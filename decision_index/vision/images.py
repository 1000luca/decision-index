import base64
import binascii
import io
from pathlib import Path

PASS_THROUGH = ("PNG", "JPEG", "WEBP", "GIF")


def load(ref, root=None):
    if isinstance(ref, (bytes, bytearray)):
        return bytes(ref)
    if ref.startswith("data:"):
        return base64.b64decode(ref.split(",", 1)[1])
    for p in ([Path(root) / ref] if root else []) + [Path(ref)]:
        try:
            if p.is_file():
                return p.read_bytes()
        except OSError:
            pass
    try:
        return base64.b64decode(ref, validate=True)
    except (binascii.Error, ValueError):
        raise FileNotFoundError(ref)


def pil(ref, root=None):
    from PIL import Image

    im = Image.open(io.BytesIO(load(ref, root)))
    im.load()
    return im


def data_url(ref, root=None):
    from PIL import Image

    raw = load(ref, root)
    with Image.open(io.BytesIO(raw)) as im:
        if im.format in PASS_THROUGH:
            return f"data:image/{im.format.lower()};base64," + base64.b64encode(raw).decode()
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def jpeg_data_url(ref, root=None, max_pixels=None, quality=90):
    from PIL import Image

    im = pil(ref, root).convert("RGB")
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def encode(ref, how, root=None):
    if how == "data_url":
        return data_url(ref, root)
    if how == "base64":
        return data_url(ref, root).split(",", 1)[1]
    if how == "path":
        return str((Path(root) / ref) if root and not Path(ref).is_absolute() else Path(ref))
    raise ValueError(f"unknown image encoding {how!r}; use data_url, base64 or path")
