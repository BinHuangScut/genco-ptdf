# Figure export and visual QA

All four final PNG previews were visually inspected. Axis labels, panel labels,
legends, data marks and uncertainty ranges are readable and do not overlap.
The error-distribution and runtime figures use dedicated shared-legend rows;
the runtime error axis starts at zero. Negative relative errors are not shown.

Figures are generated solely with Python/matplotlib. Each figure is exported as
PNG, editable-text SVG, vector PDF and 600 dpi LZW TIFF. Width is 183 mm and
height is at most 159 mm. SVG text checks and exact dimensions are recorded in
`results/figures/export_checks.json`.

Source CSVs accompany each figure. Captions in the English report distinguish
600-point transaction evaluation from the 32-point complete-basis subset and
three-seed variability from operating-point variability. The training comparison
also supplies common-update-prefix validation curves; final test checkpoints use
independent early stopping. Runtime panels identify the derivative-only classical
baseline and the separate PF-plus-derivative baseline.

Observed limitations are retained: IEEE14 higher-load transaction error regresses,
complete-basis improvements are not consistent, and single-point neural derivative
queries are slower than the sparse classical baseline on this hardware.
