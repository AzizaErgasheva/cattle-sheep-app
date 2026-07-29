import io

import numpy as np
import pytest
from PIL import Image

from app.domain.entities import ImageInput
from app.domain.exceptions import InvalidImageError
from app.infrastructure.image_utils import PillowThumbnailer, decode_and_preprocess


def _make_png_bytes(size: tuple[int, int] = (50, 40), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_and_preprocess_shapes_and_scales():
    image = ImageInput(content=_make_png_bytes(), filename="test.png")

    arr = decode_and_preprocess(image, img_size=(224, 224))

    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_decode_and_preprocess_raises_on_invalid_bytes():
    image = ImageInput(content=b"not an image", filename="bad.png")

    with pytest.raises(InvalidImageError):
        decode_and_preprocess(image, img_size=(224, 224))


def test_pillow_thumbnailer_returns_jpeg_data_url_within_bounds():
    image = ImageInput(content=_make_png_bytes(size=(500, 300)), filename="big.png")

    data_url = PillowThumbnailer().to_data_url(image, size=(96, 96))

    assert data_url.startswith("data:image/jpeg;base64,")


def test_pillow_thumbnailer_raises_on_invalid_bytes():
    image = ImageInput(content=b"garbage", filename="bad.png")

    with pytest.raises(InvalidImageError):
        PillowThumbnailer().to_data_url(image, size=(96, 96))
