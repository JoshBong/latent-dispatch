# Milestone paper

ACM two-column milestone draft: `main.tex` is the shared source and
`references.bib` is the bibliography. The draft compiles to three pages.
Before submission, confirm team responsibilities and author order, fill Josh
and Ajit's net IDs, and review the scientific claims and planned work.
Elliott's net ID is filled in as `esk226`.

## Compile locally

On Linux x86_64 (Python 3 and Make required):

```sh
cd paper
make setup  # downloads checksum-verified Tectonic 0.17.0 into ignored tools/
make pdf
```

On macOS:

```sh
brew install tectonic
cd paper
make pdf
```

The output is `paper/build/main.pdf`. Tectonic downloads missing LaTeX packages
and fonts on the first build and caches them; initial compilation needs internet.
It automatically reruns LaTeX and BibTeX as needed. No Python ML dependencies or
GPU are required. With Tectonic already installed, setup can be skipped.
On Windows, install Tectonic and run `tectonic --keep-logs --outdir build main.tex`
from this directory after creating `build`.

A full TeX Live or MiKTeX installation is also supported:
`latexmk -pdf -outdir=build main.tex` (install latexmk and ACM's dependencies).

## Collaborating

Edit the source on your own branch and open a pull request. Commit `.tex`, `.bib`,
and figure sources; build files and the local compiler are ignored. Before
submission, finish/remove TODOs and change `\drafttrue` to `\draftfalse`.
The bibliography includes LatentMAS, Parallel Context Windows, Masking in
Multi-hop QA, and MuSiQue, with in-text citations. The tables are preliminary;
keep the scoring and screening limitations explicit.

## Overleaf

Upload this directory's source files as a project and select `main.tex`.
`acmart.cls` and `ACM-Reference-Format.bst` are included, so no separate template
upload is necessary. The ACM title/authors/abstract occupy the first page;
there is no separate cover page. Aim for 2–3 substantive pages after writing.

## Pinned ACM template

The unmodified ACM `acmart.cls` v2.20 and `ACM-Reference-Format.bst` are vendored
from https://github.com/borisveytsman/acmart/tree/v2.20 to match the assignment.
The class was generated from the unmodified `vendor/acmart/acmart.dtx` using
`acmart.ins` with LaTeX. Original source/install files and `ACM-LICENSE.txt`
are included for provenance and redistribution under LPPL 1.3c.
Do not change margins or fonts to force a page count.

## Result provenance

The draft uses the repository journal through `cb1e31c` (November 16):
- Main EM table and original source-level intervals: 8B masking archive,
  `Qwen3-8B-bfloat16-masking-2026-11-15.tar.gz`.
- Small thinking EM/F1 table: 25 sources per dataset from the paper_v1 run,
  commit `116151f`; no independent-replication claim.
- F1 and screening sensitivity: the November 16 evening journal entry,
  grouped bootstrap over sources sharing a supporting paragraph. Screening
  is retrospective and single-reader, so it is presented as sensitivity analysis.
- 4B text-report observation: the completed thinking-25 branch and latest
  journal analysis. No claim of a controlled model-size trend.

No new inference was run while writing the milestone. Future experiments are
proposals for team review, not completed work. Two other authors' net IDs and
the team-contribution paragraph remain unverified.
