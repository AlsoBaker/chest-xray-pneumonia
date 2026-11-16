"""
lung_mask.py — crop a chest X-ray down to the lungs.

Uses torchxrayvision's pretrained PSPNet (ChestX-Det) to segment the left and
right lungs, cleans each lung up (largest component + convex hull, so dense
pneumonia consolidation isn't cut out), dilates slightly, and blacks out
everything else: ECG leads, side markers, text, shoulders and image borders.

The same function must be used for training AND inference.
"""
import numpy as np
import cv2
import torch
import torchxrayvision as xrv

SEG_SIZE = 512
_model = None


def _seg_model():
    global _model
    if _model is None:
        _model = xrv.baseline_models.chestx_det.PSPNet()
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _model


def _clean(m):
    """Largest connected component, then its convex hull."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(m, np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(comp)
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cnts))], 1)
    return hull


def lung_masks(gray_batch):
    """
    gray_batch: list of HxW uint8 grayscale images (any size).
    Returns list of (mask HxW uint8 {0,1}, ok: bool).
    ok=False means segmentation found too little lung; mask is then all ones.
    """
    model = _seg_model()
    dev = next(model.parameters()).device
    x = np.stack([
        cv2.resize(g, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
        for g in gray_batch
    ]).astype(np.float32)
    x = xrv.datasets.normalize(x, 255)[:, None]          # -> [-1024, 1024], Nx1xHxW
    with torch.no_grad():
        out = model(torch.from_numpy(x).to(dev))
    li, ri = model.targets.index("Left Lung"), model.targets.index("Right Lung")
    probs = torch.sigmoid(out[:, [li, ri]]).cpu().numpy()

    results = []
    pad = max(3, SEG_SIZE // 40)                          # ~2.5% margin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    for g, p in zip(gray_batch, probs):
        m = _clean(p[0] > 0.5) | _clean(p[1] > 0.5)
        m = cv2.dilate(m, kernel)
        ok = m.mean() > 0.12                              # lungs are ~25-45% of the frame
        if not ok:
            m = np.ones_like(m)
        h, w = g.shape
        results.append((cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST), ok))
    return results


def mask_xray(gray, size):
    """Grayscale uint8 X-ray -> lung-only uint8 image of size x size, plus ok flag."""
    sq = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    (m, ok), = lung_masks([sq])
    return (sq * m).astype(np.uint8), ok
