# Figure contract

Backend: Python/matplotlib only. Archetype: quantitative grid. Final width 183 mm,
height 100–160 mm. Editable SVG/PDF text, 600 dpi TIFF and 220 dpi PNG previews.
All panels use real held-out results. No synthetic result data or selective removal.

1. **Error distributions:** quantify the change in AC-PTDF error across the test
   and higher-load distributions. Four panels are grid × distribution. Empirical
   CDFs retain tails; each operating-point value averages the errors of the three
   fine-tuning seeds (not an ensemble derivative). DC and initialization have one
   value per point. The classical result defines the reference.
2. **Flow versus derivative error:** show the observed association rather than
   infer derivatives from state accuracy. Two scatter panels, one per grid, show
   initialization and seed-0 fine-tuned models on both test distributions.
   Descriptive Spearman coefficients are computed per model. This is not a causal
   test or a theorem that state and derivative errors are unrelated.
3. **Fine-tuning comparison:** compare PF and PF+response using paired training
   seeds. Four panels are grid × distribution; two additional panels show paired
   validation learning curves through each seed's common observed update budget.
   Each final-test dot is one seed's mean over
   all held-out operating points; bars show seed mean and error bars show the
   minimum–maximum range. Paired lines connect the same seed. n=3 independent
   training seeds; no hypothesis test or significance stars. Shared-prefix validation
   results support the budget comparison, but do not replace a matched-update
   held-out checkpoint experiment; individual early stopping remains a limitation.
4. **Accuracy and cost:** two panels locate seed-0 methods in accuracy–latency
   space, and two show per-point query latency for batches 1,16,64. Classical
   derivative-only and PF+derivative costs are separate. Query counts and dtype
   differ by grid and are recorded in the timing CSV. Full-matrix timings are
   delivered separately. GPU synchronization, warm-up, and 20 repeats are used.

Metrics: per-point relative Frobenius error ||H-H*||F/||H*||F; PTDF MAE in MW/MW;
flow MAE in MW. Sources: results/<grid>/per_point.csv, summary.csv, timing.csv.
Source tables are copied to results/figures/source_data. Results on the full
non-reference injection basis are additionally reported on 16 test and 16 stress
points per grid, without claiming this subset represents all 600 test points.

Review risks: conditional fixed-control derivatives include control-limit
violations; generator dispatch is scaled rather than OPF-optimized; initialization
uses a changed reference-voltage boundary mask; GPU atomic reductions are not
bitwise reproducible; ±0.1/±1 MW response fitting need not guarantee a better
zero-step derivative. Legends and the write-up must state these limitations.

QA: inspect all four PNGs at final size; check axes, legends, text overlap, line
visibility, editable vector text, source provenance, sample counts, and exports.
