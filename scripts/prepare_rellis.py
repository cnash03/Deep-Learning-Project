"""Fetch only the RELLIS-3D camera frames we need (default: the test split).

The full image archive (Rellis_3D_pylon_camera_node.zip) is 11.6 GB. Google
Drive serves it with HTTP range support, so this script reads the zip's
central directory remotely and pulls just the members listed in the chosen
split files. Labels (94 MB zip) and split lists (44 KB) are downloaded whole.

Output layout (what src/data/rellis.py reads):

    data/rellis/<seq>/pylon_camera_node/<frame>.jpg
    data/rellis/<seq>/pylon_camera_node_label_id/<frame>.png
    data/rellis/split/{train,val,test}.lst

Usage:
    .venv/bin/python scripts/prepare_rellis.py --root data/rellis --splits test
    .venv/bin/python scripts/prepare_rellis.py --root data/rellis --splits test --min-free-gb 10

Re-running skips files that already exist. Stops if free disk drops below
--min-free-gb.
"""
from __future__ import annotations

import argparse
import io
import shutil
import struct
import sys
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
IDS = {
    "images": "1F3Leu0H_m6aPVpZITragfreO_SGtL2yV",   # Rellis_3D_pylon_camera_node.zip, 11.6 GB
    "labels": "16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5",   # Rellis_3D_pylon_camera_node_label_id.zip, 94 MB
    "split": "1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ",    # Rellis_3D_image_split.zip, 44 KB
    "ontology": "1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6",  # Rellis_3D_ontology.zip, 18 KB
}
PREFIX = "Rellis-3D/"  # every archive member path starts with this
UA = {"User-Agent": "python-urllib"}


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP Range requests."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos = url, block, 0
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])
        self._cache: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _get(self, start: int, end: int) -> bytes:  # inclusive end
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) == end - start + 1:
                    return data
            except Exception as e:  # noqa: BLE001
                print(f"  range retry {attempt + 1}: {e}", file=sys.stderr)
        raise IOError(f"range {start}-{end} failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n >= self.block:  # large member reads: fetch directly
            data = self._get(self.pos, self.pos + n - 1)
        else:  # small reads (headers, central dir): block cache
            out = bytearray()
            p = self.pos
            while len(out) < n:
                b = p // self.block
                if b not in self._cache:
                    if len(self._cache) > 64:
                        self._cache.clear()
                    s = b * self.block
                    self._cache[b] = self._get(s, min(s + self.block, self.size) - 1)
                chunk = self._cache[b][p - b * self.block:]
                take = chunk[: n - len(out)]
                out += take
                p += len(take)
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def download(file_id: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {dest.name}")
        req = urllib.request.Request(DRIVE.format(file_id), headers=UA)
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    return dest


def wanted_paths(root: Path, splits: list[str]) -> tuple[set[str], set[str]]:
    imgs, labs = set(), set()
    for s in splits:
        for line in (root / "split" / f"{s}.lst").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                imgs.add(parts[0])
                labs.add(parts[1])
    return imgs, labs


def extract(zf: zipfile.ZipFile, wanted: set[str], root: Path, min_free: float, label: str) -> int:
    members = {i.filename[len(PREFIX):]: i for i in zf.infolist() if i.filename.startswith(PREFIX)}
    missing = wanted - members.keys()
    if missing:
        print(f"WARNING: {len(missing)} {label} entries not in archive, e.g. {sorted(missing)[:3]}")
    todo = sorted(p for p in wanted & members.keys() if not (root / p).exists())
    need = sum(members[p].file_size for p in todo) / 1e9
    print(f"{label}: {len(wanted)} wanted, {len(todo)} to fetch ({need:.2f} GB), free {free_gb(root):.1f} GB")
    if free_gb(root) - need < min_free:
        sys.exit(f"Refusing: would leave < {min_free} GB free")
    for k, rel in enumerate(todo):
        if k % 100 == 0:
            if free_gb(root) < min_free:
                sys.exit(f"Stopping: free disk below {min_free} GB")
            print(f"  {label} {k}/{len(todo)}")
        out = root / rel  # rel comes from the split list and matched an archive member exactly
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        with zf.open(members[rel]) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.rename(out)
    return len(todo)


def _fetch_member(remote: HTTPRangeFile, info: zipfile.ZipInfo) -> bytes:
    """One ranged GET per member: local header + generous slack for name/extra + data."""
    start = info.header_offset
    end = min(remote.size, start + 30 + 1024 + info.compress_size) - 1
    buf = remote._get(start, end)
    if buf[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    n_name, n_extra = struct.unpack("<HH", buf[26:30])
    off = 30 + n_name + n_extra
    if off + info.compress_size > len(buf):  # unusually large extra field
        buf += remote._get(end + 1, start + off + info.compress_size - 1)
    data = buf[off:off + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise IOError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(data) != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return data


def extract_remote(remote, infos, wanted: set[str], root: Path, min_free: float, workers: int) -> None:
    members = {i.filename[len(PREFIX):]: i for i in infos if i.filename.startswith(PREFIX)}
    missing = wanted - members.keys()
    if missing:
        print(f"WARNING: {len(missing)} image entries not in archive, e.g. {sorted(missing)[:3]}")
    todo = sorted(p for p in wanted & members.keys() if not (root / p).exists())
    need = sum(members[p].file_size for p in todo) / 1e9
    print(f"images: {len(wanted)} wanted, {len(todo)} to fetch ({need:.2f} GB), free {free_gb(root):.1f} GB",
          flush=True)
    if free_gb(root) - need < min_free:
        sys.exit(f"Refusing: would leave < {min_free} GB free")

    def job(rel: str) -> None:
        out = root / rel  # rel came from the split list and matched an archive member exactly
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(_fetch_member(remote, members[rel]))
        tmp.rename(out)

    with ThreadPoolExecutor(workers) as ex:
        for k, fut in enumerate(as_completed([ex.submit(job, r) for r in todo])):
            fut.result()
            if k % 100 == 0:
                print(f"  images {k}/{len(todo)}", flush=True)
                if free_gb(root) < min_free:
                    ex.shutdown(cancel_futures=True)
                    sys.exit(f"Stopping: free disk below {min_free} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/rellis")
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--workers", type=int, default=16, help="parallel range requests for images")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    zips = root / "_zips"
    zips.mkdir(parents=True, exist_ok=True)

    # split lists ("<seq>/pylon_camera_node/x.jpg <seq>/pylon_camera_node_label_id/x.png"), already in our format
    with zipfile.ZipFile(download(IDS["split"], zips / "split.zip")) as z:
        for n in ("train.lst", "val.lst", "test.lst"):
            (root / "split").mkdir(exist_ok=True)
            (root / "split" / n).write_bytes(z.read(n))
    with zipfile.ZipFile(download(IDS["ontology"], zips / "ontology.zip")) as z:
        (root / "ontology.csv").write_bytes(z.read("Rellis_3D_ontology/ontology.csv"))

    imgs, labs = wanted_paths(root, args.splits)
    with zipfile.ZipFile(download(IDS["labels"], zips / "label_id.zip")) as z:
        extract(z, labs, root, args.min_free_gb, "labels")

    remote = HTTPRangeFile(DRIVE.format(IDS["images"]))
    print(f"image archive: {remote.size / 1e9:.2f} GB (reading selectively)")
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16)) as z:
        infos = z.infolist()  # only the central directory is fetched here
    extract_remote(remote, infos, imgs, root, args.min_free_gb, args.workers)

    if not args.keep_zips:
        shutil.rmtree(zips)
    print(f"done; free disk {free_gb(root):.1f} GB")


if __name__ == "__main__":
    main()
