"""CUDA graph execution of the unchanged PF and response objectives."""
import torch


def objective(solver, batch, selection, denominator, truth, mode, weight):
    output, flows = solver.forward(batch)
    target = batch["bus"].y[:, 3:5]
    state = torch.mean((output["bus"][selection] - target[selection]) ** 2)
    keys = sorted(solver.model.layer_residuals)
    raw = [0.5 ** (len(keys) - i - 1) for i in range(len(keys))]
    physics = sum(solver.model.layer_residuals[k] * (w / sum(raw)) for k, w in zip(keys, raw))
    response = torch.zeros((), device=solver.device)
    if mode == "response":
        size = len(denominator)
        slope = (flows[size:2 * size] - flows[2 * size:]) / denominator
        response = torch.mean((slope - truth) ** 2)
    return 0.9 * state + 0.1 * physics + weight * response, state, physics, response


class CapturedStep:
    def __init__(self, solver, optimizer, batch, denominator, truth, mode, weight):
        self.solver, self.optimizer = solver, optimizer
        self.batch = batch.clone()
        self.denominator, self.truth = denominator.clone(), truth.clone()
        self.selection = self.batch.mask_dict["bus"][:, 3:5].nonzero(as_tuple=True)
        self.mode, self.weight = mode, weight
        initial = [p.detach().clone() for p in solver.model.parameters()]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.losses = self._step()
        # Capture/warm-up executes updates. Restore both weights and Adam's
        # moments, so these preparatory calls are not training steps.
        with torch.no_grad():
            for parameter, value in zip(solver.model.parameters(), initial):
                parameter.copy_(value)
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()
        torch.cuda.synchronize()

    def _step(self):
        self.optimizer.zero_grad(set_to_none=True)
        losses = objective(self.solver, self.batch, self.selection, self.denominator,
                           self.truth, self.mode, self.weight)
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(self.solver.model.parameters(), 1.0)
        self.optimizer.step()
        return losses

    def run(self, batch, denominator, truth):
        for kind in ("bus", "gen"):
            self.batch[kind].x.copy_(batch[kind].x)
            self.batch[kind].y.copy_(batch[kind].y)
        self.denominator.copy_(denominator)
        self.truth.copy_(truth)
        self.graph.replay()
        return self.losses
