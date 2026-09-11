# UAI poster draft

This folder contains a first poster draft for **Prior-Fitted Functional Flows: In-Context Generative Models for Pharmacokinetics**.

The layout follows the general three-column structure of the earlier AISTATS poster, while all scientific text, results, authors, tables, and figures are taken from the PFF paper. The `figures/` directory contains only figures copied from the paper source. Institutional logos are kept separately in `logos/`.

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error pff_poster.tex
```

The generated poster size is 185 cm × 90 cm, matching the existing poster template.
