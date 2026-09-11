"""Turning the model's internal summary into column 12.

The model describes each image with 64 numbers. SCHEMA.md froze column 12
at 32. This file does that squashing -- and, more importantly it freezes it.

Why freezing matters
    The squashing picks the 32 directions along which images differ most,
    and it works those out by looking at a sample of images. A different
    sample picks different directions.

    Picture surveying a city and fixing a coordinate grid: where north is,
    where the origin sits, how long one unit is. Every address you record
    afterwards means something only relative to that grid. If someone
    quietly re-surveys on Wednesday and rotates north thirty degrees, every
    address filed on Monday now points somewhere else. Nothing looks
    broken. The numbers are all still there. They just mean different
    things depending on when they were written.

    So: fit once, save to data/projection.npz, commit it, and load that
    file forever after. A point logged on Day 1 and a point logged on Day 3
    then sit in the same space, which is the only reason P1 can compare
    them.

Whitening
    On by default, and it is not cosmetic. Raw components have wildly
    different spreads -- the first is several times wider than the last --
    so a uniform grid over them produces cells stretched into a sliver.
    Whitening rescales every axis to roughly unit spread, which is what
    makes P1's cell_size mean the same thing on every axis.

Fit it once:
    python -m victim.embed fit --cifar
    python -m victim.embed fit --images path/to/folder

Then check P1's cell settings against real data:
    python -m victim.embed diagnose --db data/fake.db
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EMBED_DIM = 32          # column 12 width, frozen by SCHEMA.md
DEFAULT_ARTIFACT = "data/projection.npz"


class Projection:
    """The frozen survey grid: 64 numbers in, 32 out."""

    def __init__(self, mean, components, scale=None, meta=None):
        self.mean = np.asarray(mean, dtype=np.float32)              # (64,)
        self.components = np.asarray(components, dtype=np.float32)  # (32, 64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)
        self.meta = meta or {}

    @property
    def whitened(self) -> bool:
        return self.scale is not None

    def __call__(self, feats) -> np.ndarray:
        """(N, 64) -> (N, 32). Also accepts a single (64,) vector."""
        f = np.atleast_2d(np.asarray(feats, dtype=np.float32))
        if f.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"expected {self.mean.shape[0]} features, got {f.shape[1]}")
        z = (f - self.mean) @ self.components.T
        if self.scale is not None:
            z = z / self.scale
        return z.astype(np.float32)

    def one(self, feats) -> np.ndarray:
        """Single vector -> (32,). What the API calls per request."""
        return self(feats)[0]

    # ---------------------------------------------------------- persistence

    def save(self, path=DEFAULT_ARTIFACT) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components,
                 scale=np.array([]) if self.scale is None else self.scale)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path=DEFAULT_ARTIFACT) -> "Projection":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Fit it once with: python -m victim.embed fit --cifar"
            )
        d = np.load(path)
        scale = d["scale"]
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(d["mean"], d["components"],
                   None if scale.size == 0 else scale, meta)

    # ---------------------------------------------------------- fitting

    @classmethod
    def fit(cls, feats, dim=EMBED_DIM, whiten=True, meta=None) -> "Projection":
        from sklearn.decomposition import PCA

        f = np.asarray(feats, dtype=np.float32)
        if len(f) < dim:
            raise ValueError(f"need at least {dim} samples to fit {dim} directions")
        pca = PCA(n_components=dim, random_state=0).fit(f)
        scale = np.sqrt(pca.explained_variance_).astype(np.float32) if whiten else None
        info = dict(meta or {})
        info.update({
            "n_samples": int(len(f)),
            "in_dim": int(f.shape[1]),
            "out_dim": int(dim),
            "whitened": bool(whiten),
            "variance_kept": float(pca.explained_variance_ratio_.sum()),
        })
        return cls(pca.mean_, pca.components_, scale, info)


# ------------------------------------------------------------------ fitting data

def _features_from_images(paths, backend="auto", limit=5000, batch=64):
    """Run images through the victim and collect its 64-number summaries."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from victim.loader import predict

    out, buf = [], []
    for p in list(paths)[:limit]:
        buf.append(Path(p).read_bytes())
        if len(buf) == batch:
            out.append(predict(buf, backend)[1])
            buf = []
    if buf:
        out.append(predict(buf, backend)[1])
    if not out:
        raise ValueError("no images found")
    return np.concatenate(out, axis=0)


def _features_from_cifar(backend="auto", limit=5000, batch=128):
    """The right sample to fit on: the kind of pictures the API will see.

    Downloads the CIFAR-10 test set once into data/. P2 and P4 need it too.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("  importing torchvision (slow on a cold start)...", flush=True)
    try:
        import torchvision
    except ImportError as exc:
        raise SystemExit(
            "torchvision is not installed, and CIFAR-10 loading needs it.\n"
            "  pip install torchvision==0.20.1\n"
            "Or fit on your own images instead:\n"
            "  python -m victim.embed fit --images path/to/folder"
        ) from exc

    from victim.loader import load_victim, normalize

    print("  fetching CIFAR-10 test set into data/ (~170MB on first run)...",
          flush=True)
    ds = torchvision.datasets.CIFAR10(root="data", train=False, download=True)

    print(f"  loading the victim model (backend={backend})...", flush=True)
    model = load_victim(backend)

    print(f"  running {min(limit, len(ds))} images through it...", flush=True)

    imgs = np.stack([
        np.transpose(np.asarray(ds[i][0], dtype=np.float32) / 255.0, (2, 0, 1))
        for i in range(min(limit, len(ds)))
    ])
    out = [model.forward(normalize(imgs[i:i + batch]))[1]
           for i in range(0, len(imgs), batch)]
    return np.concatenate(out, axis=0)


# ------------------------------------------------------------------ diagnostics

def diagnose(points, cell_sizes=(0.2, 0.5, 1.0, 2.0), dims=(2, 4, 6, 8, 12, 16, 32)):
    """How P1's grid behaves on real points, per cell_size and dimension count.

    The number that matters is "unique cells as a fraction of queries". If it
    sits near 100%, every query lands somewhere new, coverage never
    saturates, and every client looks like it is revealing fresh ground --
    the Stage 6 signal does not exist. Somewhere well below 100% is where
    the map behaves like a map.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n = len(pts)
    rows = []
    for k in dims:
        if k > pts.shape[1]:
            continue
        sub = pts[:, :k]
        for cs in cell_sizes:
            idx = np.round(sub / cs).astype(np.int64)
            uniq = len(np.unique(idx, axis=0))
            rows.append({"dims": k, "cell_size": cs, "unique_cells": uniq,
                         "fraction_unique": uniq / n})
    return rows


def axis_spread(points) -> np.ndarray:
    """Standard deviation per axis. Near 1.0 everywhere means whitening worked."""
    return np.atleast_2d(np.asarray(points, dtype=np.float64)).std(axis=0)


# ------------------------------------------------------------------ CLI

def _cmd_fit(a) -> None:
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(
            f"{out} already exists.\n\n"
            "Refitting moves the survey grid. Every point logged under the old\n"
            "one becomes incomparable to every point logged under the new one --\n"
            "silently, with no error, because the numbers still look fine. P1's\n"
            "map would be measuring two different spaces at once.\n\n"
            "If you are sure, pass --force AND delete every log written under the\n"
            "old grid. Otherwise leave it alone: this file is committed so the\n"
            "whole team measures from the same origin."
        )

    print("fitting the projection -- this takes a few minutes the first time\n",
          flush=True)
    if a.cifar:
        feats = _features_from_cifar(a.backend, a.limit)
        source = "cifar10-test"
    elif a.images:
        paths = sorted(
            p for p in Path(a.images).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        feats = _features_from_images(paths, a.backend, a.limit)
        source = str(a.images)
    else:
        raise SystemExit("give --cifar or --images FOLDER")

    proj = Projection.fit(feats, dim=a.dim, whiten=not a.no_whiten,
                          meta={"source": source, "backend": a.backend})
    proj.save(a.out)

    pts = proj(feats)
    print(f"fitted on {len(feats)} images from {source}")
    print(f"  {feats.shape[1]} -> {a.dim} numbers, whitened={proj.whitened}")
    print(f"  variance kept: {proj.meta['variance_kept']:.1%}")
    print(f"  saved to {a.out}  -- COMMIT THIS FILE, then never refit")
    sd = axis_spread(pts)
    print(f"  spread per axis: min {sd.min():.2f}, max {sd.max():.2f} "
          f"({'even -- good' if sd.max() / sd.min() < 2 else 'uneven'})")

    print("\nhow P1's grid behaves on these points:")
    _print_diagnosis(diagnose(pts))


def _cmd_diagnose(a) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from api.logstore import LogStore

    df = LogStore(a.db).read_df(run_id=a.run_id)
    pts = np.stack(df["embedding"].dropna().to_numpy())
    print(f"{len(pts)} points from {a.db} (run_id={a.run_id})")
    sd = axis_spread(pts)
    print(f"spread per axis: min {sd.min():.2f}, max {sd.max():.2f}\n")
    _print_diagnosis(diagnose(pts))


def _print_diagnosis(rows) -> None:
    print(f"  {'dims':>5} {'cell_size':>10} {'cells':>9} {'unique':>8}   verdict")
    for r in rows:
        f = r["fraction_unique"]
        verdict = ("every query is new -- NO SIGNAL" if f > 0.90 else
                   "too fine" if f > 0.60 else
                   "usable" if f > 0.05 else
                   "too coarse -- everything collides")
        print(f"  {r['dims']:>5} {r['cell_size']:>10} {r['unique_cells']:>9} "
              f"{f:>7.1%}   {verdict}")
    print("\n  'usable' rows are where the map behaves like a map: clients")
    print("  revisit ground, coverage saturates, and a client sweeping the")
    print("  whole space stands out. Hand these to P1 for cells.py.")


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="The frozen 64 -> 32 projection")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit once and freeze")
    f.add_argument("--cifar", action="store_true", help="fit on the CIFAR-10 test set")
    f.add_argument("--images", help="fit on a folder of images instead")
    f.add_argument("--backend", default="auto", choices=["auto", "torch", "stub"])
    f.add_argument("--limit", type=int, default=5000)
    f.add_argument("--dim", type=int, default=EMBED_DIM)
    f.add_argument("--no-whiten", action="store_true")
    f.add_argument("--out", default=DEFAULT_ARTIFACT)
    f.add_argument("--force", action="store_true",
                   help="overwrite an existing projection (see the warning)")
    f.set_defaults(func=_cmd_fit)

    d = sub.add_parser("diagnose", help="check P1's cell settings against logged points")
    d.add_argument("--db", default="data/fake.db")
    d.add_argument("--run-id", default="cal_seed1")
    d.set_defaults(func=_cmd_diagnose)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    _main()
