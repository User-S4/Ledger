"""Input Validation. Stage 9, and a quarter of the score.

Everything that decides whether a request is acceptable lives here, so
there is one place to test and one place to change. main.py calls
`extract_image_bytes`, loader.py calls `decode_image_array`.

Two rules shape all of it:

  Reject what is not an image. Truncated files, text renamed to .png, empty
  bodies, three megabytes of zeroes.

  Accept images that are merely inconvenient. A 4000x3000 photo, a
  grayscale JPEG, something with transparency. Honest customers send
  whatever their phone produced. Rejecting those is a false positive
  dressed up as validation, and false positives are the number that decides
  whether any of this is deployable.

Every rejection raises ValidationError carrying a short machine tag, which
becomes column 8 in the log, and an HTTP status. Nothing ever propagates a
stack trace: a traceback leaks file paths, library versions and directory
layout, which is free reconnaissance for whoever sent the bad input.
"""

from __future__ import annotations

import base64
import binascii
import io
import json

import numpy as np

IMAGE_SIZE = 32

# A small file can decode to an enormous bitmap -- the "decompression bomb".
# A 4000x3000 photo is 12M pixels and legitimate. 89M is not.
MAX_PIXELS = 40_000_000

# Rejections. The tag becomes column 8; the status is what the client sees.
ERRORS = {
    "empty_body":        (400, "no image in the request"),
    "bad_json":          (400, "body was not valid JSON"),
    "bad_base64":        (400, "image_b64 was not valid base64"),
    "bad_image":         (400, "could not decode the image"),
    "image_too_large":   (400, "image dimensions are unreasonably large"),
    "payload_too_large": (413, "request body exceeds the limit"),
}


class ValidationError(Exception):
    """A rejection with a machine tag and an HTTP status."""

    def __init__(self, code: str):
        self.code = code
        self.status, self.detail = ERRORS.get(code, (400, "invalid request"))
        super().__init__(code)


# ------------------------------------------------------------------ body

async def extract_image_bytes(request, max_upload: int) -> bytes:
    """Pull the raw image bytes out of a request.

    Accepts a multipart upload or {"image_b64": "..."}. We dispatch on
    Content-Type by hand rather than declaring both a File and a Body
    parameter, because FastAPI cannot have both on one route -- declaring
    File() forces the whole endpoint to expect multipart.
    """
    ctype = (request.headers.get("content-type") or "").lower()

    if ctype.startswith("multipart/form-data"):
        return await _from_multipart(request, max_upload)

    body = await request.body()
    # Base64 inflates by about a third; allow for that plus JSON overhead.
    if len(body) > max_upload * 4 // 3 + 1024:
        raise ValidationError("payload_too_large")
    if not body:
        raise ValidationError("empty_body")

    if ctype.startswith("application/json"):
        return _from_json(body, max_upload)

    return body   # raw bytes under some other content type


async def _from_multipart(request, max_upload: int) -> bytes:
    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001 -- malformed multipart
        raise ValidationError("bad_image") from exc

    upload = form.get("file")
    if upload is None or isinstance(upload, str):
        raise ValidationError("empty_body")

    raw = await upload.read()
    if len(raw) > max_upload:
        raise ValidationError("payload_too_large")
    if not raw:
        raise ValidationError("empty_body")
    return raw


def _from_json(body: bytes, max_upload: int) -> bytes:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError("bad_json") from exc

    if not isinstance(payload, dict) or "image_b64" not in payload:
        raise ValidationError("empty_body")

    b64 = payload["image_b64"]
    if not isinstance(b64, str) or not b64:
        raise ValidationError("bad_image")

    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("bad_base64") from exc

    if len(raw) > max_upload:
        raise ValidationError("payload_too_large")
    if not raw:
        raise ValidationError("empty_body")
    return raw


# ------------------------------------------------------------------ image

def decode_image_array(raw: bytes, size: int = IMAGE_SIZE) -> np.ndarray:
    """Raw bytes -> float32 array (3, size, size), values 0..1.

    Resizes rather than rejecting anything that is genuinely an image.
    """
    from PIL import Image

    if not raw:
        raise ValidationError("empty_body")

    try:
        img = Image.open(io.BytesIO(raw))
        # Header only so far. Check the declared dimensions before load()
        # allocates a bitmap for them -- otherwise a 200-byte file can ask
        # us to allocate several gigabytes.
        w, h = img.size
        if w * h > MAX_PIXELS or w <= 0 or h <= 0:
            raise ValidationError("image_too_large")
        img.load()
    except ValidationError:
        raise
    except Image.DecompressionBombError as exc:
        # Pillow has its own bomb guard that fires inside open(), before our
        # size check gets a look. Same rejection, but keep the specific
        # reason so column 8 records the real cause.
        raise ValidationError("image_too_large") from exc
    except Exception as exc:  # noqa: BLE001 -- any decode failure
        raise ValidationError("bad_image") from exc

    try:
        img = img.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("bad_image") from exc

    if arr.shape != (size, size, 3):
        raise ValidationError("bad_image")
    return np.transpose(arr, (2, 0, 1)).copy()