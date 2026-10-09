"""Matched-data and maximum-budget PF versus local-response fine-tuning."""
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .classical import Grid, OperatingPoint, ROOT
from .data import Dataset
from .neural import GencoSolver
from .capture import CapturedStep


def validation_score(solver, dataset, config):
    solver.model.eval()
    indices = dataset.indices("val")[:config["validation_examples"]]
    D = dataset.arrays["D"][:, :config["validation_directions"]]
    weights = np.linalg.lstsq(solver.grid.transfer_matrix(), D, rcond=None)[0]
    errors = []
    validation_batch = config.get("validation_batch_size", 64)
    for start in range(0, len(indices), validation_batch):
        ids = indices[start:start + validation_batch]
        predicted = solver.directional_batch([dataset.point(i) for i in ids], D,
                                              capture=config.get("cuda_graph", False))
        truth = dataset.arrays["H"][ids] @ weights
        errors.extend(np.linalg.norm(predicted - truth, axis=(1, 2)) / np.linalg.norm(truth, axis=(1, 2)))
    return float(np.mean(errors))


def train(name, mode, seed, config):
    folder = ROOT / "checkpoints/finetuned" / name / mode / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        print(f"Already complete: {name}/{mode}/seed{seed}", flush=True)
        return
    settings = config["training"]
    if settings.get("dtype", "float32") != "float32":
        raise ValueError("The reference training configuration uses float32")
    torch.set_num_threads(settings["torch_threads"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    dataset = Dataset(name)
    grid = Grid(name)
    solver = GencoSolver(grid)
    captured = settings.get("cuda_graph", False) and solver.device == "cuda"
    optimizer = torch.optim.AdamW(solver.model.parameters(), lr=settings["learning_rate"],
                                 capturable=captured, fused=captured)
    stepper = None
    a = dataset.arrays
    train_ids = dataset.indices("train")
    history, best, stale = [], float("inf"), 0
    initial = validation_score(solver, dataset, settings)
    print(f"{name} {mode} seed{seed} initial validation {initial:.6f}", flush=True)
    # Preserve the initialization if training never improves held-out derivatives.
    torch.save(solver.model.state_dict(), folder / "best.pt")
    best = initial
    started = time.perf_counter()
    for epoch in range(settings["epochs"]):
        solver.model.train()
        order = rng.permutation(train_ids)
        epoch_losses = []
        for start in range(0, len(order), settings["batch_size"]):
            ids = order[start:start + settings["batch_size"]]
            size = len(ids)
            selected_pairs = rng.integers(0, a["pair_index"].shape[1], size)
            selected_steps = rng.integers(0, len(a["steps"]), size)
            selected_queries = a["pair_index"][ids, selected_pairs]
            steps = a["steps"][selected_steps]
            D = a["D"][:, selected_queries].T
            delta_pd = -steps[:, None] * D * grid.injection_load_weights
            delta_pg = steps[:, None] * (D @ grid.injection_gen_weights.T)
            pd_batch = np.concatenate([a["pd"][ids], a["pd"][ids] + delta_pd, a["pd"][ids] - delta_pd])
            qd_batch = np.tile(a["qd"][ids], (3, 1))
            pg_batch = np.concatenate([a["pg"][ids], a["pg"][ids] + delta_pg, a["pg"][ids] - delta_pg])
            plus_v = a["pair_V"][ids, selected_pairs, selected_steps, 0]
            minus_v = a["pair_V"][ids, selected_pairs, selected_steps, 1]
            targets = np.concatenate([a["V"][ids], plus_v, minus_v])
            batch = solver.batch(pd_batch, qd_batch, pg_batch, targets)
            denominators = torch.tensor(2 * steps[:, None], device=solver.device, dtype=solver.dtype)
            true_pairs = a["pair_flow"][ids, selected_pairs, selected_steps]
            true_slope = (true_pairs[:, 0] - true_pairs[:, 1]) / (2 * steps[:, None])
            truth = torch.tensor(true_slope, device=solver.device, dtype=solver.dtype)
            if captured:
                if size != settings["batch_size"]:
                    raise ValueError("CUDA graph training requires complete batches")
                if stepper is None:
                    stepper = CapturedStep(solver, optimizer, batch, denominators, truth,
                                           mode, settings["response_weight"])
                loss, state, physics, response = stepper.run(batch, denominators, truth)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss: {name}/{mode}/seed{seed}")
                epoch_losses.append([float(x.detach()) for x in (loss, state, physics, response)])
                continue
            optimizer.zero_grad(set_to_none=True)
            output, flows = solver.forward(batch)
            pf, state, physics = solver.pf_loss(output, batch)
            response = torch.zeros((), device=solver.device)
            if mode == "response":
                slope = (flows[size:2*size] - flows[2*size:]) / denominators
                response = torch.mean((slope - truth) ** 2)
            loss = pf + settings["response_weight"] * response
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {name}/{mode}/seed{seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(solver.model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append([float(loss.detach()), float(state.detach()), float(physics.detach()), float(response.detach())])
        score = validation_score(solver, dataset, settings)
        losses = np.mean(epoch_losses, axis=0)
        improved = score < best
        if improved:
            best, stale = score, 0
            torch.save(solver.model.state_dict(), folder / "best.pt")
        else:
            stale += 1
        row = {"epoch": epoch + 1, "loss": losses[0], "state": losses[1], "physics": losses[2],
               "response": losses[3], "validation_ptdf_relative": score,
               "elapsed_seconds": time.perf_counter() - started, "improved": improved}
        history.append(row)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"{name} {mode} seed{seed} epoch {epoch+1}: val={score:.5f}, best={best:.5f}, "
              f"loss={losses[0]:.5g}, elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        if stale >= settings["patience"]:
            break
    torch.save(solver.model.state_dict(), folder / "last.pt")
    completion = {"grid": name, "mode": mode, "seed": seed, "config": settings,
                  "initial_validation": initial, "best_validation": best, "epochs": len(history),
                  "training_seconds": time.perf_counter() - started,
                  "selection": "mean analytic AD PTDF relative error on fixed held-out validation directions",
                  "best_epoch": max([r["epoch"] for r in history if r["improved"]], default=0),
                  "training_data_sha256": dataset.manifest["dataset_sha256"],
                  "execution": "CUDA graph (validated equivalent objective and AdamW update)" if captured else "eager",
                  "effective_graph_batch": "8 base groups; base,+h,-h each; 24 graphs at full batch",
                  "matched_pf_control": "same groups, perturbations, ordering and update budget for each seed"}
    (folder / "complete.json").write_text(json.dumps(completion, indent=2))
    del stepper, optimizer, solver, dataset
    import gc
    gc.collect()
    torch.cuda.empty_cache()
