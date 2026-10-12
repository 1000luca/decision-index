import base64
import binascii
import io
from pathlib import Path

PASS_THROUGH = ("PNG", "JPEG", "WEBP", "GIF")
MAX_PIXELS = 1600 * 32 * 32


def fit(w, h, max_pixels):
    if not max_pixels or w * h <= max_pixels:
        return w, h
    s = (max_pixels / (w * h)) ** 0.5
    w2, h2 = max(1, int(w * s)), max(1, int(h * s))
    while w2 * h2 > max_pixels:
        if w2 >= h2:
            w2 -= 1
        else:
            h2 -= 1
    return w2, h2


def capped(ref, root=None, max_pixels=MAX_PIXELS):
    from PIL import Image

    raw = load(ref, root)
    with Image.open(io.BytesIO(raw)) as im:
        size = fit(im.width, im.height, max_pixels)
        if size == im.size:
            return raw
        fmt = "PNG" if im.format == "PNG" else "JPEG"
        out = im.convert("RGBA" if fmt == "PNG" and ("A" in im.mode or "transparency" in im.info) else "RGB").resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return buf.getvalue()


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


def pil(ref, root=None, max_pixels=None):
    from PIL import Image

    im = Image.open(io.BytesIO(capped(ref, root, max_pixels) if max_pixels else load(ref, root)))
    im.load()
    return im


def data_url(ref, root=None, max_pixels=None):
    from PIL import Image

    raw = capped(ref, root, max_pixels) if max_pixels else load(ref, root)
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
        im = im.resize(fit(w, h, max_pixels), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def encode(ref, how, root=None, max_pixels=None):
    if how == "data_url":
        return data_url(ref, root, max_pixels)
    if how == "base64":
        return data_url(ref, root, max_pixels).split(",", 1)[1]
    if how == "path":
        path = (Path(root) / ref) if root and not Path(ref).is_absolute() else Path(ref)
        if max_pixels:
            from PIL import Image

            with Image.open(path) as im:
                if im.width * im.height > max_pixels:
                    raise ValueError(f"{path} is over {max_pixels} pixels; use data_url or base64 so the kit can downscale it")
        return str(path)
    raise ValueError(f"unknown image encoding {how!r}; use data_url, base64 or path")
