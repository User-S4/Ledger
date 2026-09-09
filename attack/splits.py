"""
Who is allowed to use which photos.

CIFAR-10 ships 50,000 training photos and 10,000 exam photos. If two people
use the same ones for different jobs, a measurement quietly becomes worthless
and nobody notices until Day 3. So the ranges are decided once, here, and
everyone imports from this file instead of slicing the dataset themselves.

  train[    0 :  5000]  P3/P1  reference photos that lay out the grid
  train[ 5000 : 35000]  P2     attacker query pool          <-- yours
  train[35000 : 50000]  P4     honest customer traffic
  test [    0 : 10000]  nobody queries these -- the fidelity exam

Move a boundary here and tell the team. Never slice around it locally.
"""

import numpy as np
from torchvision import datasets

SPLITS = {
    "grid_reference": (0, 5000),      # P3 + P1
    "attacker": (5000, 35000),        # P2
    "honest": (35000, 50000),         # P4
}


def _uint8(ds):
    return np.asarray(ds.data, dtype=np.uint8)


def load_split(name, n=None, root="./data", seed=0, download=True):
    """Photos from one owner's range. No labels -- a thief has none."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; pick from {list(SPLITS)}")
    lo, hi = SPLITS[name]
    images = _uint8(datasets.CIFAR10(root=root, train=True, download=download))[lo:hi]
    if n is None:
        return images
    if n > len(images):
        raise ValueError(f"split '{name}' holds {len(images)} photos, asked for {n}")
    idx = np.random.default_rng(seed).permutation(len(images))[:n]
    return images[idx]


def load_surrogate(n, root="./data", seed=0, download=True):
    """CIFAR-100: same size and style, completely different subjects.

    This is the honest attacker's photo album -- a pile of pictures of things
    the victim was never trained to recognise. No overlap with CIFAR-10 at all,
    so it sidesteps the split question entirely.
    """
    images = _uint8(datasets.CIFAR100(root=root, train=True, download=download))
    idx = np.random.default_rng(seed).permutation(len(images))[:n]
    return images[idx]


def load_exam(root="./data", download=True):
    """The 10,000 test photos, with true labels. Query set for nobody.

    Used only at the end, to ask the victim and the clone the same questions
    and count how often they said the same thing.
    """
    ds = datasets.CIFAR10(root=root, train=False, download=download)
    return _uint8(ds), np.asarray(ds.targets, dtype=np.int64)
