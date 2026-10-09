import argparse
from pathlib import Path
import yaml
from .classical import ROOT


def main():
    parser = argparse.ArgumentParser(description="GENCO AC-PTDF experiments")
    parser.add_argument("command", choices=["fetch", "generate", "validate", "train", "evaluate", "benchmark", "report", "all"])
    parser.add_argument("--config", default=str(ROOT / "configs/experiment.yaml"))
    parser.add_argument("--grid", choices=["case14_ieee", "case118_ieee"])
    parser.add_argument("--mode", choices=["pf", "response"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--accuracy-only", action="store_true", help="In evaluate, postpone performance measurements")
    parser.add_argument("--classical-only", action="store_true", help="In benchmark, refresh only classical timing rows")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    grids = [args.grid] if args.grid else config["grids"]
    if args.command in ("fetch", "all"):
        from .neural import fetch_checkpoints
        fetch_checkpoints(grids)
    if args.command in ("validate", "all"):
        from .validation import validate
        validate(grids)
    if args.command in ("generate", "all"):
        from .data import generate_dataset
        for grid in grids:
            folder = ROOT / "data" / grid
            if not (folder / "manifest.json").exists():
                generate_dataset(grid, config)
    if args.command in ("train", "all"):
        from .training import train
        for grid in grids:
            for mode in ([args.mode] if args.mode else ["pf", "response"]):
                for seed in ([args.seed] if args.seed is not None else config["training"]["seeds"]):
                    train(grid, mode, seed, config)
    if args.command in ("evaluate", "all"):
        from .validation import validate_finetuned, validate_dataset
        validate_dataset(grids)
        selected = dict(config, grids=grids)
        validate_finetuned(selected)
        from .evaluation import evaluate
        for grid in grids:
            evaluate(grid, config, run_timing=not args.accuracy_only)
    if args.command == "benchmark":
        from .evaluation import benchmark
        from .data import Dataset
        for grid in grids:
            benchmark(grid, Dataset(grid), config,
                      methods=["dc", "ac_jacobian", "ac_pf_jacobian"] if args.classical_only else None)
    if args.command in ("report", "all"):
        from .report import make_report
        make_report(config)


if __name__ == "__main__":
    main()
