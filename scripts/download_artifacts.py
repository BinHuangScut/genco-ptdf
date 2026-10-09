"""Download checksum-verified release assets and extract at the repository root."""
from hashlib import sha256
from pathlib import Path
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def main():
    manifest = json.loads((ROOT / "release/manifest.json").read_text())
    destination = ROOT / "release"
    for record in manifest["assets"]:
        name = record["name"]
        if Path(name).name != name:
            raise ValueError(f"Invalid asset name: {name}")
        output = destination / name
        if not output.exists() or output.stat().st_size != record["bytes"] or digest(output) != record["sha256"]:
            print(f"Downloading {name} ({record['bytes']/2**20:.1f} MiB)", flush=True)
            request = urllib.request.Request(record["url"], headers={"User-Agent": "GENCO-PTDF/0.1.0"})
            with urllib.request.urlopen(request, timeout=180) as response, output.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
        if output.stat().st_size != record["bytes"] or digest(output) != record["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {output}")
        with tarfile.open(output, "r:gz") as bundle:
            bundle.extractall(ROOT, filter="data")
        print(f"Verified and extracted {name}", flush=True)

if __name__ == "__main__":
    main()
