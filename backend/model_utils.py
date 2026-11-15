# ============================================================
# model_utils.py
# Handles model loading, inference, and Grad-CAM generation
# ============================================================

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import base64

# ── Load model + threshold once at startup ─────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_PATH = os.path.join(BASE_DIR, "optimal_threshold.json")

print("Loading model from Hugging Face Hub...")
MODEL_PATH = hf_hub_download(
    repo_id="vijayesh684/chest-xray-densenet121",
    filename="best_model_phase2.keras"
)
model = load_model(MODEL_PATH)
print("✓ Model loaded")

with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["threshold"]
print(f"✓ Threshold loaded: {THRESHOLD:.4f}")

IMG_SIZE = 380  # EfficientNetB4 native resolution

# ── Preprocessing ───────────────────────────────────────────
def preprocess_image(img_bytes):
    """
    Convert raw image bytes → normalized tensor
    EfficientNet normalizes internally, no /255 needed
    Returns: (original_rgb, preprocessed_batch)
    """
    nparr    = np.frombuffer(img_bytes, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # EfficientNet expects [0, 255] range
    img_batch = np.expand_dims(original, 0).astype(np.float32)

    return original, img_batch

# ── Inference ───────────────────────────────────────────────
def predict(img_batch):
    """
    Run inference on preprocessed image batch.
    Returns: (score, label, confidence)
    """
    score      = float(model.predict(img_batch, verbose=0)[0][0])
    label      = "PNEUMONIA" if score > THRESHOLD else "NORMAL"
    confidence = score if score > THRESHOLD else 1 - score
    return score, label, confidence

# ── Grad-CAM ────────────────────────────────────────────────
def get_gradcam_heatmap(img_batch, explain_pneumonia):
    """
    Grad-CAM for the class the model actually predicted.

    - Targets the pre-sigmoid logit, not the probability. Near scores of 0 or 1
      the sigmoid saturates and its gradients shrink to ~1e-7, leaving noise.
    - For a NORMAL prediction it explains NORMAL (negated logit) instead of
      mapping the faint leftover evidence for pneumonia.
    """
    base_model = model.get_layer("efficientnetb4")

    # Build a sub-model from input to last conv output (top_activation)
    last_conv_layer = base_model.get_layer("top_activation")
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )
    output_layer = model.get_layer("dense_2")

    img_tensor = tf.cast(img_batch, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        # Pass through the head layers, stopping before the sigmoid
        x      = model.get_layer("global_average_pooling2d")(base_outputs)
        x      = model.get_layer("batch_normalization")(x, training=False)
        x      = model.get_layer("dense")(x)
        x      = model.get_layer("dropout")(x, training=False)
        x      = model.get_layer("dense_1")(x)
        x      = model.get_layer("dropout_1")(x, training=False)
        logit  = (tf.matmul(x, output_layer.kernel) + output_layer.bias)[:, 0]
        target = logit if explain_pneumonia else -logit

    grads        = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_outputs[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0).numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

def overlay_gradcam(original_img, heatmap, floor=0.2, max_alpha=0.55):
    """
    Overlay heatmap on original image.

    Opacity follows attention: regions below `floor` show the X-ray untouched
    instead of a uniform blue wash, and the coarse 12x12 map is smoothed so
    hotspots don't render as blocks.
    """
    h = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    h = cv2.GaussianBlur(h, (0, 0), IMG_SIZE / 40)
    if h.max() > 0:
        h = h / h.max()
    h = np.clip(h, 0, 1)

    alpha           = (np.clip((h - floor) / (1 - floor), 0, 1) * max_alpha)[..., None]
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed    = original_img * (1 - alpha) + heatmap_colored * alpha
    return superimposed.astype(np.uint8)

def numpy_to_base64(img_array):
    """Convert numpy image array to base64 string for React."""
    img_bgr    = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer  = cv2.imencode(".png", img_bgr)
    b64_string = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_string}"

# ── Main pipeline ───────────────────────────────────────────
def analyze_xray(img_bytes):
    """
    Full pipeline: bytes → prediction + Grad-CAM
    Returns dict with all results for React frontend
    """
    original, img_batch      = preprocess_image(img_bytes)
    score, label, confidence = predict(img_batch)
    heatmap                  = get_gradcam_heatmap(img_batch, label == "PNEUMONIA")
    superimposed             = overlay_gradcam(original, heatmap)

    return {
        "label"      : label,
        "confidence" : round(confidence * 100, 2),
        "score"      : round(score, 4),
        "threshold"  : round(THRESHOLD, 4),
        "original"   : numpy_to_base64(original),
        "gradcam"    : numpy_to_base64(superimposed),
        "metrics"    : {
            "accuracy"   : 93.59,
            "roc_auc"    : 0.9834,
            "sensitivity": 94.36,
            "specificity": 92.31,
            "f1_score"   : 0.9485
        }
    }
