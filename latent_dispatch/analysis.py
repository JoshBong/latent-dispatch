"""Tables for the notebooks. `df` is what results.load_results returns: one row per (question, arm name).

Differences are in EM points (x100) when metric="em", raw F1 otherwise.

The unit of every CI and test is the source question. One source question shows up as several rows: 3 seeds (the
same question in 3 distractor layouts), and several placements at one crossing level (2-hop AB and BA). Those rows
are not independent, so each source question's rows are averaged first; n = rows, sources = source questions.
"""
import numpy as np
import pandas as pd

from . import stats
from .results import META


def _scale(metric):
    return 100 if metric == "em" else 1


def _key(by, key):
    return dict(zip(by, key if isinstance(key, tuple) else (key,)))


def wide(df: pd.DataFrame, metric: str = "em", arms=()) -> pd.DataFrame:
    """One row per question, one column per arm (the metric), plus the question metadata. Arms named in `arms` get
    a column even when none of their rows are in yet (all NaN), so a partly-run experiment gives empty tables."""
    w = df.pivot_table(index="id", columns="arm", values=metric)
    w = w.reindex(columns=list(dict.fromkeys([*w.columns, *arms])))
    return w.join(df.drop_duplicates("id").set_index("id")[META])


def accuracy(df: pd.DataFrame, by=("dataset", "qtype", "crossings"), metric: str = "em") -> pd.DataFrame:
    """Mean metric per arm per group, with n = questions in the group and sources = distinct source questions."""
    by = list(by)
    t = df.pivot_table(index=by, columns="arm", values=metric, aggfunc="mean") * _scale(metric)
    t.insert(0, "n", df.groupby(by).id.nunique())
    t.insert(1, "sources", df.groupby(by).source_id.nunique())
    return t.round(1 if metric == "em" else 3)


def _per_source(g: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """a − b averaged over each source question's rows (seeds, placements): one number per source question."""
    return (g[a] - g[b]).groupby(g.source_id).mean().to_numpy()


def paired(df: pd.DataFrame, a: str, b: str, by=("dataset", "crossings"), metric: str = "em", **only) -> pd.DataFrame:
    """a − b on the questions both arms ran, per group: mean difference, 95% bootstrap CI over source questions,
    sign-flip p (= exact McNemar when every source question has one row), and the p-value Holm-adjusted across the
    rows of this table. `only` filters questions, e.g. placement=["BB", "BBB"]."""
    by, s = list(by), _scale(metric)
    w = _filter(wide(df, metric, [a, b]), only).dropna(subset=[a, b])
    out = []
    for key, g in w.groupby(by):
        d = _per_source(g, a, b)
        est, lo, hi = stats.bootstrap_ci(d)
        out.append({**_key(by, key), "n": len(g), "sources": len(d), "diff": s * est, "lo": s * lo, "hi": s * hi,
                    "p": stats.sign_flip(d)})
    t = pd.DataFrame(out)
    if len(t):
        t["p_holm"] = stats.holm(t.p.tolist())
    return t.set_index(by).round(3) if len(t) else t


def equivalent(df: pd.DataFrame, a: str, b: str, by=("dataset",), margin: float = 0.02, **only) -> pd.DataFrame:
    """TOST: is a − b (EM) within ±margin? The 90% bootstrap CI (over source questions) has to sit inside it."""
    by = list(by)
    w = _filter(wide(df, "em", [a, b]), only).dropna(subset=[a, b])
    out = []
    for key, g in w.groupby(by):
        d = _per_source(g, a, b)
        t = stats.tost(d, margin)
        out.append({**_key(by, key), "n": len(g), "sources": len(d), "diff": 100 * t["diff"], "lo_90": 100 * t["lo"],
                    "hi_90": 100 * t["hi"], "equivalent": t["equivalent"]})
    return pd.DataFrame(out).set_index(by).round(2) if out else pd.DataFrame()


def crossing_trend(df: pd.DataFrame, a: str, b: str, dataset: str, base_placements=None,
                   metric: str = "em") -> pd.DataFrame:
    """Does the a − b gap grow with crossings? For each crossing level c: gap(c) − gap(0), paired on the same source
    question where both levels exist. A source question's rows at one level (its seeds; 2-hop AB and BA are both 1
    crossing) contribute their mean. base_placements restricts level 0, e.g. to BB-type."""
    w = wide(df, metric, [a, b]).dropna(subset=[a, b])
    w = w[w.dataset == dataset].assign(gap=lambda x: x[a] - x[b])
    base = w[w.crossings == 0]
    if base_placements is not None:
        base = base[base.placement.isin(base_placements)]
    g0 = base.groupby("source_id").gap.mean()
    out = []
    for c in sorted(set(w.crossings) - {0}):
        gc = w[w.crossings == c].groupby("source_id").gap.mean()
        both = g0.index.intersection(gc.index)
        if len(both):
            est, lo, hi = stats.interaction(g0[both].to_numpy(), gc[both].to_numpy(), paired=True)
            s = _scale(metric)
            out.append({"crossings": c, "sources": len(both), "gap_minus_base": s * est, "lo": s * lo, "hi": s * hi})
    return pd.DataFrame(out).set_index("crossings").round(2) if out else pd.DataFrame()


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Is every arm's output usable? One row per arm; read it after every smoke run and before trusting a table.

        boxed %          answers with a \\boxed{} (the format every prompt asks for)
        cut %            answers that hit max_answer_tokens before the model stopped. That cap should never bind,
                         so this should be ~0; if not, raise the cap and rerun that arm
        think cut %      forced-thinking answers whose thinking ran to its budget (H4's variable: reported, not a bug)
        notes / reports cut %   scan notes / T1 reports that hit their token cap before finishing
    """
    def pct(s):
        s = pd.Series(s).dropna()
        return round(100 * s.astype(float).mean(), 1) if len(s) else np.nan

    def each(col):                                   # list-valued column -> one value per element
        return [x for xs in col.dropna() for x in xs]

    rows = []
    for arm, g in df.groupby("arm", sort=False):
        forced = g.options.map(lambda o: isinstance(o, dict) and o.get("thinking") == "forced") if "options" in g \
            else pd.Series(False, index=g.index)
        rows.append({
            "arm": arm, "n": len(g),
            "boxed %": pct(g.text.str.contains("\\boxed{", regex=False)),
            "cut %": pct(g["cut"]) if "cut" in g else np.nan,
            "think cut %": pct(g.think_cut[forced]) if "think_cut" in g else np.nan,
            "notes cut %": pct(each(g["notes_cut"])) if "notes_cut" in g else np.nan,
            "reports cut %": pct(each(g["truncated"])) if "truncated" in g else np.nan,
            "empty %": pct(g.text.str.strip() == ""),
            "EM": round(100 * g.em.mean(), 1),
            "gen tokens mean": round(g.gen_tokens.mean(), 1), "gen tokens max": int(g.gen_tokens.max()),
        })
    return pd.DataFrame(rows).set_index("arm") if rows else pd.DataFrame()


def _filter(w: pd.DataFrame, only: dict) -> pd.DataFrame:
    for field, allowed in only.items():
        allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        w = w[w[field].isin(allowed)]
    return w


def peek(df: pd.DataFrame, n: int = 2, chars: int = 300) -> None:
    """Print a few raw outputs per arm. Read these before launching a full run."""
    for arm, g in df.groupby("arm", sort=False):
        for _, r in g.head(n).iterrows():
            print(f"--- {arm}  {r.id}  gold={r.answers[0]!r}  pred={r.pred!r}  em={r.em:.0f}")
            print("    " + str(r.text)[-chars:].replace("\n", "\n    "))
