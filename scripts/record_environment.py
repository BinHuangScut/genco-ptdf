import importlib.metadata
import json
import platform
import subprocess
import torch
from genco_ptdf.classical import ROOT

def record():
    packages = ("torch", "torch-geometric", "torch-scatter", "torch-sparse", "pandapower", "numpy", "scipy", "pandas")
    commits = {name: subprocess.check_output(["git", "-C", str(ROOT / "vendor" / name), "rev-parse", "HEAD"], text=True).strip()
               for name in ("gridfm-graphkit", "gridfm-datakit")}
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": subprocess.check_output(["lscpu"], text=True),
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
            "packages": {p: importlib.metadata.version(p) for p in packages}, "commits": commits,
            "reproducibility": "fixed data/config/source; GPU atomic reductions may not be bitwise deterministic"}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/environment.json").write_text(json.dumps(info, indent=2))

if __name__ == "__main__":
    record()
