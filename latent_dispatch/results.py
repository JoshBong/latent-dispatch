"""Run an arm over questions and save one JSON line per question; load saved rows back as a scored table.

    results/<model>-<dtype>/<name>.jsonl      e.g. results/Qwen3-4B-bfloat16/k0.jsonl

`name` is the arm plus its options, chosen by the notebook ("k1_m8" = arms.k1 with m=8). Every notebook that asks
for the same name reuses the same rows, so H3 doesn't rerun the K0 that H1 already ran. Runs resume: questions
already in the file are skipped. Each row stores the options and the git commit; asking for a name with different
options raises instead of mixing.

If you change code in latent_dispatch/ that changes what an arm generates, delete that arm's results file (or give
the new version a new name). Scoring changes need no rerun: load_results re-scores every row from its raw text.
"""
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from .data import ROOT
from .score import score

RESULTS = ROOT / "results"
META = ["dataset", "qtype", "placement", "crossings", "hops", "seed", "source_id"]


def results_path(model_name: str, dtype: str, name: str) -> Path:
    return RESULTS / f"{model_name.split('/')[-1]}-{dtype}" / f"{name}.jsonl"


def _read(path: Path) -> dict[str, dict]:
    """Saved rows by question id. A line cut off by a kill mid-write is skipped (that question just reruns)."""
    if not path.exists():
        return {}
    rows, bad = {}, 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                rows[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                bad += 1
    if bad:
        print(f"warning: skipped {bad} unreadable line(s) in {path.name}")
    return rows


def _check(path: Path, rows: dict, want: dict) -> None:
    """Every saved row must come from the same arm and options the notebook asks for."""
    for r in rows.values():
        if {"arm": r.get("arm"), "options": r.get("options")} != want:
            raise ValueError(f"{path} holds {r.get('arm')} {r.get('options')}, not {want['arm']} {want['options']}. "
                             "Pick another name, or delete the file to rerun.")


def _commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(model, questions: list[dict], name: str, arm, **options) -> None:
    """Run `arm(model, q, **options)` on every question not already saved under `name`."""
    path = results_path(model.name, model.dtype_name, name)
    done = _read(path)
    want = {"arm": arm.__name__, "options": json.loads(json.dumps(options))}
    _check(path, done, want)
    todo = [q for q in questions if q["id"] not in done]
    where = f"results/{path.parent.name}/{path.name}"
    print(f"{name}: {len(todo)} to run, {len(questions) - len(todo)} already saved  ({where})")
    if not todo:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    commit, t_start = _commit(), time.perf_counter()
    with path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            t0 = time.perf_counter()
            out = arm(model, q, **options)
            row = {"id": q["id"], **want, "model": model.name, "dtype": model.dtype_name, "commit": commit,
                   "seconds": round(time.perf_counter() - t0, 3), **out}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.perf_counter() - t_start) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s each, ~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)


def load_results(questions: list[dict], arms: dict, model_name: str, dtype: str) -> pd.DataFrame:
    """One row per (question, arm name) that has been run, re-scored from the raw text with the current scorer,
    with the question's metadata (dataset, qtype, placement, crossings, ...) and answers/bridges joined on.

    `arms` is the notebook's {name: (arm function, options)}, so rows made with other options are refused here too."""
    by_id = {q["id"]: q for q in questions}
    rows = []
    for name, (arm, options) in arms.items():
        path = results_path(model_name, dtype, name)
        saved = _read(path)
        _check(path, saved, {"arm": arm.__name__, "options": json.loads(json.dumps(options))})
        mine = [saved[i] for i in by_id if i in saved]
        commits = sorted({r["commit"] for r in mine})
        print(f"{name}: {len(mine)}/{len(by_id)} questions" + (f"  (commits {commits})" if len(commits) > 1 else ""))
        for r in mine:
            q = by_id[r["id"]]
            pred, em, f1 = score(r["text"], q["answers"])
            rows.append({**r, "arm": name, "fn": r["arm"], "pred": pred, "em": float(em), "f1": f1,
                         **{k: q[k] for k in META}, "answers": q["answers"], "bridges": q["bridges"]})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["id", "arm", "text", "pred", "em", "f1", *META])
