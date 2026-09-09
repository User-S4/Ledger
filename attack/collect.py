"""
The robbery: send photos, write down every answer.

One output only -- a .npz holding (photos, answers). That is the thief's
notebook, and it is what train_clone.py learns from.

Note what is NOT here: a query log. SCHEMA.md says only the API writes log
rows, and it's right. If the attacker wrote its own logs they'd be a record of
what we claim we sent, not evidence of what actually arrived. Once you point
this at P3's real endpoint, the logs appear on their side automatically and
P1 reads them from there.

The clone trains on `returned_probs` -- what the customer actually received --
never on `model_probs`. That distinction is the entire point of Stage 7: when
the defence starts degrading answers, the thief only ever sees the degraded
ones, and the copy gets worse.
"""

from pathlib import Path

import numpy as np


def collect(service, images, batch_size=512, out_npz=None, verbose=True):
    images = np.asarray(images, dtype=np.uint8)
    response = service.predict(images, batch_size=batch_size)
    answers = np.asarray(response["returned_probs"], dtype=np.float32)

    if out_npz:
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            images=images,
            answers=answers,
            degradation_level=response.get("degradation_level", 0),
        )
        if verbose:
            print(f"    stole {len(images)} photo/answer pairs -> {out_npz}")

    return answers


def load_stolen(path):
    d = np.load(path)
    return d["images"], d["answers"]
