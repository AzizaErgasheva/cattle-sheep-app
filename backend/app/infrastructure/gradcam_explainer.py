"""Concrete ExplainerPort implementation: renders a Grad-CAM overlay PNG.

Reuses the fixed graph-construction approach from the training notebook to
handle both flat models (e.g. a custom CNN) and models that wrap a nested
backbone submodel (ResNet50 / MobileNetV2 / EfficientNetB0). See
`_heatmap()` for why the nested case needs two separate graphs instead of one
`tf.keras.Model` spanning both.

TensorFlow is imported lazily for the same reason as in keras_classifier.py.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image

from app.domain.entities import ImageInput
from app.domain.exceptions import ModelNotLoadedError
from app.domain.ports import ExplainerPort
from app.infrastructure.image_utils import decode_and_preprocess

_PREPROCESS_FN_NAMES = {"resnet50", "mobilenet", "efficientnet"}


class GradCamAdapter(ExplainerPort):
    def __init__(
        self,
        model_path: str | Path,
        img_size: tuple[int, int],
        preprocess_fn_name: str | None = None,
    ):
        if preprocess_fn_name is not None and preprocess_fn_name not in _PREPROCESS_FN_NAMES:
            raise ValueError(f"Unknown preprocess_fn_name: {preprocess_fn_name!r}")
        self._model_path = Path(model_path)
        self._img_size = img_size
        self._preprocess_fn_name = preprocess_fn_name
        self._model = None
        self._tf = None

    def _load(self):
        if self._model is not None:
            return self._model
        if not self._model_path.exists():
            raise ModelNotLoadedError(f"Model file not found at {self._model_path}")
        import tensorflow as tf  # local import -- see module docstring
        self._tf = tf
        self._model = tf.keras.models.load_model(self._model_path)
        return self._model

    def _last_conv_layer_name(self, model) -> str:
        tf = self._tf
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if isinstance(layer, tf.keras.Model):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        return sub.name
        raise ValueError("No Conv2D layer found in model")

    def _base_submodel(self, model):
        tf = self._tf
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                return layer
        return None

    def _preprocess_fn(self):
        tf = self._tf
        return {
            "resnet50": tf.keras.applications.resnet50.preprocess_input,
            "mobilenet": tf.keras.applications.mobilenet_v2.preprocess_input,
            "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
        }.get(self._preprocess_fn_name, lambda x: x)

    def _heatmap(self, model, img_batch) -> np.ndarray:
        tf = self._tf
        conv_name = self._last_conv_layer_name(model)
        base = self._base_submodel(model)

        if base is None:
            # Flat architecture -- conv layer sits directly on `model`.
            grad_model = tf.keras.models.Model(
                model.inputs, [model.get_layer(conv_name).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_batch, training=False)
                tape.watch(conv_out)
                loss = preds[:, 0]
            grads = tape.gradient(loss, conv_out)
        else:
            # Nested backbone -- build a self-contained graph from the backbone's
            # own input/output, then replay the head manually in the same tape.
            conv_layer = base.get_layer(conv_name)
            base_grad_model = tf.keras.Model(base.input, [conv_layer.output, base.output])

            head_layers, seen_base = [], False
            for layer in model.layers:
                if layer is base:
                    seen_base = True
                    continue
                if seen_base and not isinstance(layer, tf.keras.layers.InputLayer):
                    head_layers.append(layer)

            preprocess_fn = self._preprocess_fn()
            with tf.GradientTape() as tape:
                x = preprocess_fn(img_batch * 255.0)
                conv_out, base_out = base_grad_model(x, training=False)
                tape.watch(conv_out)
                h = base_out
                for layer in head_layers:
                    h = layer(h, training=False)
                loss = h[:, 0]
            grads = tape.gradient(loss, conv_out)

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    @staticmethod
    def _colorize(heatmap: np.ndarray) -> np.ndarray:
        """Maps a [0,1] heatmap to an RGB 'hot' colormap, no matplotlib dependency."""
        h = np.clip(heatmap, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4 * h - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * h - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * h - 1), 0, 1)
        return np.stack([r, g, b], axis=-1)

    def explain(self, image: ImageInput) -> bytes:
        model = self._load()
        tf = self._tf
        img_batch = decode_and_preprocess(image, self._img_size)
        heatmap = self._heatmap(model, tf.convert_to_tensor(img_batch))

        heatmap_rgb = (self._colorize(heatmap) * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_rgb).resize(self._img_size)

        base_rgb = (img_batch[0] * 255).astype(np.uint8)
        base_img = Image.fromarray(base_rgb).convert("RGB")

        overlay = Image.blend(base_img, heatmap_img.convert("RGB"), alpha=0.45)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return buf.getvalue()
