# Milestone paper

ACM two-column milestone scaffold: `main.tex` is the shared source and
`references.bib` is the bibliography. Fill in the red TODOs and the three net
IDs; confirm author order. This is not a completed milestone submission.

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
The bibliography includes the three starting papers; add actual in-text
citations when writing the related-work section. The existing results table is preliminary and
uses the automatic scorer; keep its limitations explicit.

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
