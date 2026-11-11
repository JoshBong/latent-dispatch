"""Tables for the notebooks. `df` is what results.load_results returns: one row per (question, arm name).

Differences are in EM points (x100) when metric="em", raw F1 otherwise.
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
    """Mean metric per arm per group, with n = questions in the group."""
    by = list(by)
    t = df.pivot_table(index=by, columns="arm", values=metric, aggfunc="mean") * _scale(metric)
    t.insert(0, "n", df.groupby(by).id.nunique())
    return t.round(1 if metric == "em" else 3)


def paired(df: pd.DataFrame, a: str, b: str, by=("dataset", "crossings"), metric: str = "em", **only) -> pd.DataFrame:
    """a − b on the questions both arms ran, per group: mean difference, 95% bootstrap CI, exact McNemar p, and the
    p-value Holm-adjusted across the rows of this table. `only` filters questions, e.g. placement=["BB", "BBB"]."""
    by, s = list(by), _scale(metric)
    w = _filter(wide(df, metric, [a, b]), only).dropna(subset=[a, b])
    out = []
    for key, g in w.groupby(by):
        est, lo, hi = stats.bootstrap_ci((g[a] - g[b]).to_numpy())
        p = stats.mcnemar(g[a], g[b])["p"] if metric == "em" else np.nan
        out.append({**_key(by, key), "n": len(g), "diff": s * est, "lo": s * lo, "hi": s * hi, "p": p})
    t = pd.DataFrame(out)
    if metric == "em" and len(t):
        t["p_holm"] = stats.holm(t.p.tolist())
    return t.set_index(by).round(3) if len(t) else t


def equivalent(df: pd.DataFrame, a: str, b: str, by=("dataset",), margin: float = 0.02, **only) -> pd.DataFrame:
    """TOST: is a − b (EM) within ±margin? The 90% bootstrap CI has to sit inside the margin."""
    by = list(by)
    w = _filter(wide(df, "em", [a, b]), only).dropna(subset=[a, b])
    out = []
    for key, g in w.groupby(by):
        t = stats.tost((g[a] - g[b]).to_numpy(), margin)
        out.append({**_key(by, key), "n": len(g), "diff": 100 * t["diff"], "lo_90": 100 * t["lo"],
                    "hi_90": 100 * t["hi"], "equivalent": t["equivalent"]})
    return pd.DataFrame(out).set_index(by).round(2) if out else pd.DataFrame()


def crossing_trend(df: pd.DataFrame, a: str, b: str, dataset: str, base_placements=None,
                   metric: str = "em") -> pd.DataFrame:
    """Does the a − b gap grow with crossings? For each crossing level c: gap(c) − gap(0), paired on the same
    question (source_id, seed) where both levels exist. A question with several placements at one level (2-hop AB
    and BA are both 1 crossing) contributes the mean over them. base_placements restricts level 0, e.g. to BB-type."""
    w = wide(df, metric, [a, b]).dropna(subset=[a, b])
    w = w[w.dataset == dataset].assign(gap=lambda x: x[a] - x[b])
    base = w[w.crossings == 0]
    if base_placements is not None:
        base = base[base.placement.isin(base_placements)]
    g0 = base.groupby(["source_id", "seed"]).gap.mean()
    out = []
    for c in sorted(set(w.crossings) - {0}):
        gc = w[w.crossings == c].groupby(["source_id", "seed"]).gap.mean()
        both = g0.index.intersection(gc.index)
        if len(both):
            est, lo, hi = stats.interaction(g0[both].to_numpy(), gc[both].to_numpy(), paired=True)
            s = _scale(metric)
            out.append({"crossings": c, "n_paired": len(both), "gap_minus_base": s * est, "lo": s * lo, "hi": s * hi})
    return pd.DataFrame(out).set_index("crossings").round(2) if out else pd.DataFrame()


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
