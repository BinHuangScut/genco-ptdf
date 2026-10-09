"""Source-grounded figures and an English challenge submission draft."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .classical import ROOT

LABELS = {"dc": "DC", "ac_jacobian": "AC Jacobian", "ac_pf_jacobian": "AC PF + Jacobian",
          "official_init": "GENCO initialization", "pf": "PF fine-tune", "response": "PF + response"}
COLORS = {"dc": "#6C737B", "ac_jacobian": "#2A3239", "ac_pf_jacobian": "#A2A7AD",
          "official_init": "#7F9DBB", "pf": "#3F8D9F", "response": "#A96E9F"}
MARKERS = {"dc": "s", "ac_jacobian": "*", "official_init": "^", "pf": "o", "response": "D"}
METHODS = ["dc", "official_init", "pf", "response"]


def markdown_table(frame):
    columns = frame.columns.tolist()
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))


def make_report(config):
    folder = ROOT / "results/figures"
    source = folder / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    frames, summaries, timings, full = [], [], [], []
    for name in config["grids"]:
        for filename, collection in (("per_point.csv", frames), ("summary.csv", summaries),
                                     ("timing.csv", timings), ("full_matrix_per_point.csv", full)):
            path = ROOT / "results" / name / filename
            collection.append(pd.read_csv(path))
            shutil.copyfile(path, source / f"{name}_{filename}")
    points, summary, timing, full = [pd.concat(x, ignore_index=True) for x in (frames, summaries, timings, full)]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.7, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    exports = []
    def save(fig, name):
        for extension in ("svg", "pdf", "png", "tiff"):
            options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
            fig.savefig(folder / f"{name}.{extension}", dpi=600 if extension == "tiff" else 220, **options)
        svg = ET.parse(folder / f"{name}.svg")
        exports.append({"figure": name, "svg_text_elements": len(svg.findall(".//{http://www.w3.org/2000/svg}text")),
                        "size_inches": list(fig.get_size_inches()), "source": "source_data/", "backend": "Python/matplotlib"})
        plt.close(fig)
    def label(ax, letter):
        ax.text(-0.13, 1.09, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    def grid_label(name):
        return "IEEE 14" if "14_" in name else "IEEE 118"
    def shared_legend_grid(height):
        fig = plt.figure(figsize=(7.205, height), layout="constrained")
        layout = fig.add_gridspec(3, 2, height_ratios=[1, 1, .17])
        axes = np.array([[fig.add_subplot(layout[r, c]) for c in range(2)] for r in range(2)])
        legend = fig.add_subplot(layout[2, :])
        legend.set_axis_off()
        return fig, axes, legend

    # 1: ECDF of seed-averaged per-point errors, retaining the complete tail.
    fig, axes, legend = shared_legend_grid(5.0)
    for row, name in enumerate(config["grids"]):
        for col, split in enumerate(("test", "stress")):
            ax = axes[row, col]
            subset = points[(points.grid == name) & (points.split == split)]
            for method in METHODS:
                values = subset[subset.method == method].groupby("group").ptdf_relative.mean().sort_values().to_numpy() * 100
                ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", color=COLORS[method],
                        linestyle="--" if method == "dc" else "-", label=LABELS[method], linewidth=1.4)
            ax.set_xscale("log")
            ax.set_xlabel("Relative PTDF error (%)")
            ax.set_ylabel("Fraction of operating points")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{grid_label(name)} · {'regular' if split == 'test' else 'higher load'} (n={len(subset.group.unique())})")
            label(ax, "abcd"[row*2+col])
    handles, names = axes[0, 0].get_legend_handles_labels()
    legend.legend(handles, names, loc="center", ncols=4, frameon=False)
    save(fig, "01_error_distributions")

    # 2: no inference of causality from the plotted descriptive association.
    correlations = []
    fig, axes = plt.subplots(1, 2, figsize=(7.205, 3.1), layout="constrained")
    for ax, name, letter in zip(axes, config["grids"], "ab"):
        subset = points[(points.grid == name) & (points.seed == 0)]
        for method in METHODS[1:]:
            selected = subset[subset.method == method]
            rho = spearmanr(selected.flow_mae_mw, selected.ptdf_relative).statistic
            correlations.append({"grid": name, "method": method, "n": len(selected), "spearman_rho": rho})
            ax.scatter(selected.flow_mae_mw, selected.ptdf_relative * 100, s=7, alpha=0.45,
                       color=COLORS[method], marker=MARKERS[method], linewidths=0,
                       label=f"{LABELS[method]} (rho={rho:.2f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Base flow MAE (MW)"); ax.set_ylabel("Relative PTDF error (%)")
        ax.set_title(f"{grid_label(name)} · regular + higher load")
        ax.legend(loc="best")
        label(ax, letter)
    pd.DataFrame(correlations).to_csv(source / "flow_derivative_correlations.csv", index=False)
    save(fig, "02_flow_vs_derivative")

    # 3: paired seeds; range is variability of seeds, not an operating-point CI.
    fig, axes = plt.subplots(3, 2, figsize=(7.205, 6.25), layout="constrained")
    common_budgets = []
    curve_rows = []
    for row, name in enumerate(config["grids"]):
        for col, split in enumerate(("test", "stress")):
            ax = axes[row, col]
            selected = summary[(summary.grid == name) & (summary.split == split)]
            seed_values = {}
            for x, method in enumerate(METHODS):
                values = selected[selected.method == method].sort_values("seed").ptdf_relative.to_numpy() * 100
                center = values.mean()
                ax.bar(x, center, color=COLORS[method], alpha=0.24, width=.65, zorder=1)
                ax.errorbar(x, center, yerr=[[center-values.min()], [values.max()-center]],
                            color=COLORS[method], capsize=3, linewidth=1, zorder=3)
                offsets = np.linspace(-.1, .1, len(values)) if len(values) > 1 else [0]
                ax.scatter(x + np.asarray(offsets), values, color=COLORS[method], marker=MARKERS[method], s=18, zorder=4)
                if method in ("pf", "response"):
                    seed_values[method] = values
            for j in range(len(config["training"]["seeds"])):
                ax.plot([2-.1+.1*j, 3-.1+.1*j], [seed_values["pf"][j], seed_values["response"][j]],
                        color="#ADB0B5", linewidth=.65, zorder=2)
            ax.set_yscale("log")
            ax.set_xticks(range(4), ["DC", "Init.", "PF", "PF + response"])
            ax.set_ylabel("Mean relative PTDF error (%)")
            ax.set_title(f"{grid_label(name)} · {'regular' if split == 'test' else 'higher load'}")
            label(ax, "abcd"[row*2+col])
    for col, name in enumerate(config["grids"]):
        ax = axes[2, col]
        for seed in config["training"]["seeds"]:
            histories, initials = {}, {}
            for method in ("pf", "response"):
                run = ROOT / "checkpoints/finetuned" / name / method / f"seed{seed}"
                histories[method] = pd.read_csv(run / "history.csv")
                initials[method] = json.loads((run / "complete.json").read_text())["initial_validation"]
            common = min(len(h) for h in histories.values())
            record = {"grid": name, "seed": seed, "common_epochs": common,
                      "updates_per_epoch": int(np.ceil(config["counts"]["train"]/config["training"]["batch_size"]))}
            for method in ("pf", "response"):
                values = np.r_[initials[method], histories[method].validation_ptdf_relative.to_numpy()[:common]]
                best = np.minimum.accumulate(values)
                curve_rows.extend({"grid": name, "seed": seed, "method": method, "epoch": epoch,
                                   "validation_ptdf_relative": float(value), "best_validation_ptdf_relative": float(best[epoch])}
                                  for epoch, value in enumerate(values))
                ax.plot(np.arange(common+1), best*100, color=COLORS[method], alpha=.7,
                        linestyle=["-", "--", ":"][config["training"]["seeds"].index(seed)], linewidth=1,
                        label=LABELS[method] if seed == config["training"]["seeds"][0] else None)
                record[f"{method}_best_validation"] = float(best[-1])
            common_budgets.append(record)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch (same 250 updates per epoch)")
        ax.set_ylabel("Best validation PTDF error (%)")
        ax.set_title(f"{grid_label(name)} · common prefix per seed")
        label(ax, "ef"[col])
    axes[2, 0].legend()
    pd.DataFrame(common_budgets).to_csv(source / "common_budget_validation.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(source / "common_budget_curves.csv", index=False)
    save(fig, "03_finetuning_comparison")

    # 4: single-point tradeoff and batch throughput; derivative-only AC is explicit.
    fig, axes, legend = shared_legend_grid(5.4)
    for col, name in enumerate(config["grids"]):
        ax = axes[0, col]
        latency = timing[(timing.grid == name) & (timing.scope == "transaction_matrix") & (timing.batch_size == 1)]
        regular = summary[(summary.grid == name) & (summary.split == "test")]
        for method in ["dc", "ac_jacobian", "official_init", "pf", "response"]:
            time_value = latency[latency.method == method].seconds_per_point.iloc[0] * 1000
            score = regular[(regular.method == method) & regular.seed.isin([-1, 0])].ptdf_relative.iloc[0] * 100
            ax.scatter(time_value, score, color=COLORS[method], marker=MARKERS[method], s=38,
                       label=LABELS[method], zorder=3)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=.01)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Single-point query latency (ms)"); ax.set_ylabel("Relative PTDF error (%)")
        ax.set_title(f"{grid_label(name)} · seed 0, regular test")
        label(ax, "ab"[col])
        ax = axes[1, col]
        batch_timing = timing[(timing.grid == name) & (timing.scope == "transaction_matrix")]
        for method in ["dc", "ac_jacobian", "ac_pf_jacobian", "official_init", "pf", "response"]:
            values = batch_timing[batch_timing.method == method].sort_values("batch_size")
            ax.plot(values.batch_size, values.seconds_per_point * 1000, color=COLORS[method],
                    marker=MARKERS.get(method, "x"), linestyle="--" if method == "ac_pf_jacobian" else "-",
                    markersize=4, linewidth=1.1, label=LABELS[method])
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([1, 16, 64], ["1", "16", "64"])
        ax.set_xlabel("Operating points per batch"); ax.set_ylabel("Query latency per point (ms)")
        ax.set_title(f"{grid_label(name)} · {int(latency.transactions.iloc[0])} transfers")
        label(ax, "cd"[col])
    handles, names = axes[1, 0].get_legend_handles_labels()
    legend.legend(handles, names, loc="center", ncols=3, frameon=False)
    save(fig, "04_accuracy_and_latency")
    (folder / "export_checks.json").write_text(json.dumps(exports, indent=2))
    write_submission(config, summary, points, timing, full, correlations)
    write_manifest()


def write_manifest():
    """Refresh snapshot hashes without regenerating measured data or figures."""
    manifest = []
    for parent in ("data", "checkpoints", "results", "configs"):
        for path in sorted((ROOT / parent).rglob("*")):
            if (path.is_file() and "pilot_initial_dispatch" not in path.parts
                    and ".cache" not in path.parts and "logs" not in path.parts
                    and path.name not in ("artifact_manifest.json", "verification.txt")):
                manifest.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                                 "sha256": sha256(path.read_bytes()).hexdigest()})
    sources = [p for parent in ("genco_ptdf", "scripts", "tests", "docs")
               for p in (ROOT / parent).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".md")]
    sources += [ROOT / p for p in ("README.md", "pyproject.toml", "requirements.lock.txt", "LICENSE", "NOTICE", ".gitignore", ".gitattributes", "CITATION.cff") if (ROOT / p).exists()]
    for path in sorted(sources):
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                         "sha256": sha256(path.read_bytes()).hexdigest()})
    (ROOT / "results/artifact_manifest.json").write_text(json.dumps(manifest, indent=2))


def write_submission(config, summary, points, timing, full, correlations):
    rows, changes = [], []
    for name in config["grids"]:
        for split in ("test", "stress"):
            subset = summary[(summary.grid == name) & (summary.split == split)]
            means = {}
            for method in METHODS:
                data = subset[subset.method == method]
                errors = data.ptdf_relative.to_numpy() * 100
                means[method] = errors.mean()
                rows.append({"Grid": name.replace("case", "IEEE ").replace("_ieee", ""),
                             "Set": split, "Method": LABELS[method],
                             "PTDF relative (%)": f"{errors.mean():.3f}" + (f" ± {errors.std(ddof=1):.3f}" if len(errors)>1 else ""),
                             "PTDF MAE": f"{data.ptdf_mae.mean():.5f}",
                             "Sign agreement (%)": f"{100*data.sign_agreement.mean():.2f}",
                             "+5 MW delta MAE (MW)": f"{data.delta_mae_mw.mean():.4f}"})
            changes.append({"grid": name, "split": split, "reduction_percent": 100*(1-means["response"]/means["pf"])})
    change_text = "; ".join(f"{x['grid']} {x['split']}: {x['reduction_percent']:+.1f}%" for x in changes)
    runtime = timing[(timing.scope == "transaction_matrix") & (timing.batch_size == 1)]
    time_rows = [{"Grid": r.grid, "Method": LABELS[r.method], "ms / point": f"{r.seconds_per_point*1000:.3f}",
                  "Peak GPU MiB": f"{r.peak_gpu_mib:.1f}"} for r in runtime.itertuples()]
    metadata = []
    for name in config["grids"]:
        manifest = json.loads((ROOT / "data" / name / "manifest.json").read_text())
        groups = pd.read_csv(ROOT / "data" / name / "groups.csv")
        limits = groups.groupby("split")[["voltage_limit", "q_limit", "p_limit", "branch_limit"]].mean()*100
        for split, row in limits.iterrows():
            metadata.append({"Grid": name, "Set": split, "Voltage (%)": f"{row.voltage_limit:.1f}",
                             "Q (%)": f"{row.q_limit:.1f}", "P (%)": f"{row.p_limit:.1f}", "Branch (%)": f"{row.branch_limit:.1f}",
                             "Failed generation": str(manifest["nonconvergent"])})
    full_summary = full.groupby(["grid", "split", "method", "seed"]).ptdf_relative.mean().reset_index()
    full_summary.to_csv(ROOT / "results/figures/source_data/full_matrix_summary.csv", index=False)
    full_table = []
    for name in config["grids"]:
        for split in ("test", "stress"):
            row = {"Grid": name, "Set": split}
            for method in METHODS:
                values = full_summary[(full_summary.grid == name) & (full_summary.split == split) &
                                      (full_summary.method == method)].ptdf_relative.to_numpy() * 100
                row[LABELS[method] + " (%)"] = f"{values.mean():.3f}" + (f" ± {values.std(ddof=1):.3f}" if len(values)>1 else "")
            full_table.append(row)
    outcomes = [x["reduction_percent"] > 0 for x in changes]
    finding = ("Response supervision improves the mean held-out derivative error over ordinary PF fine-tuning in all four grid/distribution comparisons."
               if all(outcomes) else "The effect of response supervision is mixed: it does not improve every grid/distribution comparison.")
    entry_path = ROOT / "configs/submission.json"
    if entry_path.exists():
        entry = json.loads(entry_path.read_text())
        submission_header = (f"**Entrant:** {entry['entrant']} ({entry['entry_type']} entry), {entry['affiliation']}.\n"
                             f"**Contact:** {entry['contact_email']}.\n"
                             f"**Code:** {entry['repository']}.\n"
                             f"**Models, data and figure archives:** {entry['release']}.\n"
                             "See [the concise Community entry](challenge_entry.md) and [artifact downloads](release.md).")
        submission_footer = "Code, models and data are released at the URLs above. The Community entry is prepared separately from this detailed research report."
    else:
        submission_header = "**Submission status:** local reproducible research draft. Entrant/team identity, contact email and public code/model/data URLs must be supplied before submission."
        submission_footer = "Public URLs and entrant identity must be added before submitting to the challenge."
    text = f"""# GENCO-PTDF: testing and improving AC power-flow derivatives

{submission_header}

## Research question and contribution

An accurate power-flow surrogate is not automatically an accurate local sensitivity
surrogate. We evaluate the derivative of the official GENCO Small models and test
whether a small amount of local transfer-response supervision improves AC-PTDF
accuracy beyond ordinary PF fine-tuning with the same extra state labels.

{finding} Relative error reductions versus PF (positive is better): {change_text}.
These are measured outcomes on this fixed-control benchmark, not a universal
guarantee. No speedup is assumed: we benchmark a cached sparse classical method.

## GENCO and OpenGridFM usage

We use official Small/seed0 IEEE14 and IEEE118 checkpoints and their per-grid
normalization statistics, pinned to Hugging Face revision
`7014f145635f26fa29e9e003f0dc33f689d615da`. GraphKit is pinned to
`95b90eecc7f5c2c86a9dd776f0c3f40434def22f`; DataKit to
`316c4afeb1b1d048b6ccaa6135fbc54e00549f64`. We retain upstream implementation
unchanged and put the PF boundary adapter and derivative APIs in a separate package.
DataKit supplies corrected networks and admittance/branch utilities. Our uniform
load sampler and balanced proportional dispatch are an extension, rather than the
official OPF-generated dataset. Official models are trained per grid; we do not
claim unseen-topology foundation-model generalization.

## Definition and experimental design

For a transfer a→b, AC-PTDF is dP_from/dt at t=0, in MW/MW. Net injection at a
increases by t MW and at b decreases by t MW, with Qd, PQ/PV types, PV/reference
voltage setpoints, reference angle, transformer taps and phase shifts fixed.
The reference generator balances incremental AC losses. At PV source nodes the
generator mapping uses Pmax shares; other coordinates use constant-Q load changes.

Each grid has 2000/400/400 training/validation/test operating points plus 200
higher-load test points. Global regular-load multipliers are 0.8–1.2; higher-load
multipliers 1.2–1.4; independent local Pd/Qd multipliers 0.9–1.1. Scheduled Pg uses
base Pg shares normalized to demand plus a fixed 3% loss allowance; actual reference
Pg follows AC balance. A base point and all eight paired perturbation states stay
in one split. Hashes, group IDs and failed solves are saved.

Five methods are compared: DC, converged-point AC Jacobian, official GENCO
initialization with the common boundary adapter, ordinary PF fine-tuning, and PF
plus response fine-tuning. Before masking, the adapter retains given voltage
setpoints and treats reference Vm as known. It never reads solved voltage labels
as inputs. Both fine-tuning arms see identical base/perturbed state supervision,
seed-specific order, and maximum update budget. Responses use ±0.1 and ±1 MW
central secants; PF training retains official masked state and layered physics
losses. AdamW lr=1e-4, batch=8 groups (24 graphs), weight=1, max=100 epochs,
patience=10, seeds=0,1,42. The best checkpoint minimizes AD PTDF error on all 400
validation points ×4 directions. Independent early stopping can change actual
update counts. Full configurations and per-epoch histories are provided.
Figure3 also compares cumulative-best validation errors within each paired seed's
common observed update prefix. This is an equal-update validation diagnostic;
the final held-out comparison still uses each arm's own early stopping. Consequently
we do not claim the final comparison fully eliminates actual-update-count effects.

## Numerical validation

The final solver directly calls pandapower's Newton-Raphson kernel with cached
DataKit admittance matrices. Classical Jacobians are rebuilt at converged states
and factorized once per point. Stored labels were originally generated by an
equivalent sparse Newton loop; all 6,000 base states/full labels and 48,000 paired
state residuals are certified against the final solver, with results in
`results/validation/pandapower_dataset.json`. Final classical timings are remeasured.
IEEE14/118 derivatives agree with high-precision central differences. An independent
pandapower Newton solver agrees with the voltage solution. GENCO block reverse AD
agrees with directional AD and convergent double-precision model finite differences.
Finite steps can cross LeakyReLU boundaries, so we record the step sweep rather
than treating one large step as a derivative. Tests check units, direction, taps,
phase shifts, reference balancing, multiple generators, masks, target isolation,
group splits and capture/eager objective equivalence.

## Results

The transaction test covers all 400 regular and 200 higher-load operating points
per grid. Fine-tuning entries report the mean ± sample standard deviation across
three training seeds of the per-point metric averages. Sign agreement uses
|Htrue|≥1e-3. The +5 MW increment is an unseen perturbation size and excludes base
flow intercept error. The classical method supplies the reference.

{markdown_table(pd.DataFrame(rows))}

![PTDF errors](../results/figures/01_error_distributions.png)
![Flow versus derivative error](../results/figures/02_flow_vs_derivative.png)
![Fine-tuning comparison](../results/figures/03_finetuning_comparison.png)

Complete non-reference injection matrices are additionally evaluated on 16 regular
and 16 higher-load points per grid using block reverse AD. Source tables report
this subset separately; it is not the full 600-point transaction test. Flow versus
derivative correlations are descriptive and are available with the source data.
No statistical significance claim is made from three seeds.

Complete-basis relative errors (subset; mean ± seed SD):

{markdown_table(pd.DataFrame(full_table))}

## Runtime and memory

RTX 4080 SUPER 16 GB and WSL Ubuntu/Python3.12 are used. Models use float32 on GPU;
classical solvers float64 on CPU. Dynamic graph preparation and CPU-result transfer
are included; checkpoint loading and static network setup are excluded. We reuse
Y matrices, the DC LU, and each point's AC Jacobian LU across transaction columns.
GPU completion is synchronized; 20 repeated measurements follow warm-up, including
CUDA graph capture for the fixed-shape directional AD queries. Full matrices use
block reverse AD. Both allocated and reserved GPU peaks are recorded.
Derivative-only AC and PF+derivative costs are separate. Batch sizes1,16,64 and
full-matrix single-point times are in the timing source tables.

{markdown_table(pd.DataFrame(time_rows))}

![Accuracy and latency](../results/figures/04_accuracy_and_latency.png)

## Scope and limitations

This is a conditional derivative study on two fixed topologies. It does not test
line outages, Q-limit switching, inverter/control-policy responses, or robust
operational feasibility. Given fixed voltage controls, even converged scenarios
can violate generator or network limits. All such samples are retained and counted:

{markdown_table(pd.DataFrame(metadata))}

Q flags use aggregate controlled-bus Q bounds; each studied case has one generator
per generator bus. The reference Pg is recovered from balance. These flags must
be considered when interpreting deployment relevance. The table contains base-point
flag rates; all paired perturbation flags are in `data/<grid>/perturbation_limits.csv`,
and +5 MW test flags in `results/<grid>/delta_control_limits.csv`. Secant training may affect
local activation-region derivatives differently, and float32 GPU atomic reductions
are not bitwise deterministic. We preserve actual checkpoints and metrics alongside
code/config/data so observed findings remain auditable.

## Open-science package and reproduction

The package includes a pinned environment, immutable upstream/checkpoint revisions,
grouped data with hashes, training scripts/configurations, all best/last checkpoints,
validation logs, per-point and per-seed metrics, limit/nonconvergence records,
synchronized timing measurements, editable SVG/PDF figures and source CSVs.
See README.md for commands and docs/methods.md for full conventions.
`results/artifact_manifest.json` records hashes of data/models/configs/results.
Project code is Apache-2.0; upstream artifacts retain their license terms.

Reproduction: `bash scripts/bootstrap.sh`, then
`GENCO_VENV=... bash scripts/run_experiments.sh`. The default config is
`configs/experiment.yaml`. A fresh output folder regenerates all experiment data
and results; completed runs can be reused for evaluation. {submission_footer}

References: [challenge](https://huggingface.co/spaces/gridfm/genco-challenge),
[official reproduction guide](https://github.com/gridfm/gridfm-paper-genco/blob/main/repro_instructions/5.4.2_PF_on_datakit.md),
[checkpoints](https://huggingface.co/gridfm/genco-pf-datakit),
[pandapower AC solver](https://pandapower.readthedocs.io/en/stable/powerflow/ac.html).
"""
    (ROOT / "docs/submission.md").write_text(text)
    pd.DataFrame(changes).to_csv(ROOT / "results/response_improvements.csv", index=False)
    (ROOT / "results/results_table.md").write_text(markdown_table(pd.DataFrame(rows)))
    pd.DataFrame(rows).to_csv(ROOT / "results/results_table.csv", index=False)

