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
import tarfile
import time
from pathlib import Path

import pandas as pd

from . import data
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
    with path.open("rb") as f:
        for line in f:
            try:
                r = json.loads(line)
                rows[r["id"]] = r
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                bad += 1
    if bad:
        print(f"warning: skipped {bad} unreadable line(s) in {path.name}")
    return rows


def _repair_tail(path: Path) -> None:
    """Recover an interrupted final write before appending: keep a complete row or remove its partial fragment."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r+b") as f:
        end = f.seek(0, 2)
        f.seek(end - 1)
        if f.read(1) == b"\n":
            return
        start, tail = end, b""
        while start > 0 and b"\n" not in tail:
            previous = max(0, start - 65536)
            f.seek(previous)
            tail = f.read(start - previous) + tail
            start = previous
        offset = tail.rfind(b"\n") + 1
        try:
            row = json.loads(tail[offset:])
            valid = isinstance(row, dict) and "id" in row
        except (json.JSONDecodeError, UnicodeDecodeError):
            valid = False
        if valid:
            f.seek(0, 2)
            f.write(b"\n")
        else:
            f.truncate(start + offset)
            print(f"warning: removed an interrupted final row in {path.name}; that question will rerun")


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
    _repair_tail(path)
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


def load_all(model_name: str, dtype: str) -> pd.DataFrame:
    """Every saved row for one model (every arm file in its results folder), re-scored, with question metadata.
    For audits across notebooks (analysis.audit). Unlike load_results it doesn't check options: it's for checking
    that outputs are usable, not for result tables. `smoke` marks the first 2 questions of every cell, the ones a
    smoke run answers (data.per_cell), so a check can judge the same rows however many more have been saved."""
    folder = results_path(model_name, dtype, "x").parent
    qs = [q for name in ("toy", "toy_d0", "musique", "hotpotqa")
          if (data.BENCHMARK / f"{name}.jsonl").exists() for q in data.load(name)]
    by_id = {q["id"]: q for q in qs}
    smoke = {q["id"] for q in data.per_cell(qs, 2)}
    rows = []
    for path in sorted(folder.glob("*.jsonl")):
        for r in _read(path).values():
            q = by_id.get(r["id"])
            if q is None:
                continue
            pred, em, f1 = score(r["text"], q["answers"])
            rows.append({**r, "arm": path.stem, "fn": r["arm"], "pred": pred, "em": float(em), "f1": f1,
                         **{k: q[k] for k in META}, "answers": q["answers"], "smoke": r["id"] in smoke})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["id", "arm", "text", "pred", "em", "f1", *META,
                                                                 "smoke"])


# ------------------------------------------------------------------------------------- start from an earlier run

ROW_FIELDS = {"id", "arm", "options", "model", "dtype", "commit", "text", "cut"}
INPUTS = ["benchmark/SHA256SUMS", "latent_dispatch/prompts.py"]      # the questions and the prompts every arm reads


def load_archive(path, root: Path = ROOT) -> None:
    """Start from a finished run: check archive/<name>.tar.gz, then unpack its results/ and notebooks/runs/ under
    `root`. `run` then skips every question the archive already answered. A file that already exists is kept as it
    is, so a rerun on the same machine never loses rows it has added since.

    Refused, before anything is unpacked: files outside those two folders; rows that aren't what `run` writes (a
    cut-off line, a missing field, two arms or two sets of options in one file, a model that doesn't match its
    folder); rows made with other benchmark files or prompts than this checkout's, which answer other inputs."""
    path, root = Path(path), Path(root)
    with tarfile.open(path) as tar:
        files = [m for m in tar.getmembers() if not m.isdir()]
        for m in files:
            parts = Path(m.name).parts
            if not m.isfile() or ".." in parts or (parts[:1] != ("results",) and parts[:2] != ("notebooks", "runs")):
                raise ValueError(f"{path.name}: {m.name} isn't a file under results/ or notebooks/runs/")
        commits = set()
        for m in files:
            if m.name.endswith(".jsonl"):
                commits |= _check_rows(f"{path.name}: {m.name}", tar.extractfile(m).read().decode("utf-8"),
                                       Path(m.name).parent.name)
        for commit in sorted(commits):
            _same_inputs(commit, path.name)
        new = [m for m in files if not (root / m.name).exists()]
        tar.extractall(root, members=new, filter="data")
    print(f"{path.name}: unpacked {len(new)} file(s), kept {len(files) - len(new)} already on this machine  "
          f"(rows made at {', '.join(sorted(commits)) or 'no commit'})")


def _check_rows(where: str, text: str, folder: str) -> set:
    """The rows of one results file are what `run` writes, for one arm, under the right model. -> their commits."""
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"{where} line {n} isn't valid JSON (cut off?)") from None
        missing = ROW_FIELDS - set(r) if isinstance(r, dict) else ROW_FIELDS
        if missing:
            raise ValueError(f"{where} line {n} has no {sorted(missing)}")
        rows.append(r)
    kinds = {(r["arm"], json.dumps(r["options"], sort_keys=True)) for r in rows}
    if len(kinds) > 1:
        raise ValueError(f"{where} mixes arms or options: {sorted(kinds)}")
    models = {f"{str(r['model']).split('/')[-1]}-{r['dtype']}" for r in rows}
    if models - {folder}:
        raise ValueError(f"{where} holds rows of {sorted(models)}, not {folder}")
    return {r["commit"] for r in rows}


def _same_inputs(commit: str, where: str) -> None:
    """Rows made at `commit` used the benchmark files and prompts of this checkout (compared through git)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    if git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError(f"{where}: its rows were made at {commit!r}, which git can't find here (uncommitted or "
                         "unknown code), so their inputs can't be compared with this checkout's")
    changed = git("diff", "--name-only", commit, "HEAD", "--", *INPUTS).stdout.split()
    if changed:
        raise ValueError(f"{where}: its rows were made at {commit} with other {', '.join(changed)} than this "
                         "checkout, so they answer other inputs and can't be mixed with new rows")
    other = git("diff", "--name-only", commit, "HEAD", "--", "latent_dispatch").stdout.split()
    if other:
        print(f"note: {', '.join(other)} changed since {commit}; if that changes what a loaded arm generates, "
              "delete that arm's file and rerun it")
