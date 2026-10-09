"""Package the verified scientific snapshot into public release archives."""
from hashlib import sha256
from pathlib import Path
import gzip
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
entry = json.loads((ROOT / "configs/submission.json").read_text())
destination = ROOT / "release"
destination.mkdir(exist_ok=True)

def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def archive(name, folder):
    output = destination / name
    files = [p for p in sorted(folder.rglob("*")) if p.is_file()
             and ".cache" not in p.parts and "__pycache__" not in p.parts]
    with output.open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for path in files:
                    info = bundle.gettarinfo(str(path), arcname=path.relative_to(ROOT).as_posix())
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as source:
                        bundle.addfile(info, source)
    record = {"name": name, "bytes": output.stat().st_size, "sha256": digest(output),
              "files": len(files),
              "url": f"{entry['repository']}/releases/download/{entry['release_tag']}/{name}"}
    print(json.dumps(record), flush=True)
    return record

records = [archive(f"data-{name}.tar.gz", ROOT / "data" / name)
           for name in ("case14_ieee", "case118_ieee")]
records += [archive("models.tar.gz", ROOT / "checkpoints"),
            archive("figures.tar.gz", ROOT / "results/figures")]
(destination / "manifest.json").write_text(json.dumps(
    {"repository": entry["repository"], "tag": entry["release_tag"], "assets": records}, indent=2))
(destination / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['name']}\n" for r in records))
