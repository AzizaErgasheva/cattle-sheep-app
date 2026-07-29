"""Shared image decode/preprocess helper used by both the classifier and
explainer adapters. Only depends on Pillow + numpy -- no TensorFlow here."""
import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.domain.entities import ImageInput
from app.domain.exceptions import InvalidImageError
from app.domain.ports import ThumbnailerPort


def decode_and_preprocess(image: ImageInput, img_size: tuple[int, int]) -> np.ndarray:
    """Decode raw bytes into a (1, H, W, 3) float32 array scaled to [0, 1],
    matching the preprocessing used in the training notebook (resize + /255)."""
    try:
        pil_image = Image.open(io.BytesIO(image.content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageError(f"Could not decode '{image.filename}' as an image") from exc

    pil_image = pil_image.resize(img_size)
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


class PillowThumbnailer(ThumbnailerPort):
    """Renders a small JPEG data URL for the history panel."""

    def to_data_url(self, image: ImageInput, size: tuple[int, int]) -> str:
        try:
            pil_image = Image.open(io.BytesIO(image.content)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise InvalidImageError(f"Could not decode '{image.filename}' as an image") from exc

        pil_image.thumbnail(size)
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=70)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
