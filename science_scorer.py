"""
Science Scorer
Computes the scientific relevance score Si for each detected target.
Uses multi-factor visual analysis: color anomaly, texture uniqueness,
edge complexity, and contextual rarity.
This implements the 'Si' term in Ui = Si / Ei.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple

log = logging.getLogger("ScienceScorer")

# Base science weights per class (from domain knowledge)
BASE_SCORES: Dict[str, float] = {
    "rock":       0.80,
    "mineral":    1.00,
    "crystal":    0.95,
    "soil":       0.45,
    "sand":       0.35,
    "gravel":     0.50,
    "water_ice":  1.00,
    "dust":       0.25,
    "obstacle":   0.05,
    "unknown":    0.30,
}

# Mars soil average BGR (approx reddish-brown)
MARS_SOIL_BGR = np.array([40, 60, 140], dtype=np.float32)


class ScienceScorer:
    """
    Computes Si = f(base_score, color_anomaly, texture, edge_complexity)
    for a detected target patch.

    Si is normalized to [0, 1].
    """

    def __init__(self):
        self._score_history: Dict[str, list] = {}

    # ── Feature extractors ────────────────────────────────────────────────────

    def _color_anomaly(self, patch: np.ndarray) -> float:
        """
        How different is the patch from average Mars soil?
        Higher → more unusual → higher scientific value.
        """
        mean_bgr = patch.mean(axis=(0, 1)).astype(np.float32)
        dist     = float(np.linalg.norm(mean_bgr - MARS_SOIL_BGR))
        return float(np.clip(dist / 150.0, 0, 1))

    def _texture_richness(self, patch: np.ndarray) -> float:
        """
        LBP-like texture richness via Laplacian variance.
        High variance = rich texture = geologically interesting.
        """
        gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        lap   = cv2.Laplacian(gray, cv2.CV_64F)
        score = float(np.clip(lap.var() / 1000.0, 0, 1))
        return score

    def _edge_complexity(self, patch: np.ndarray) -> float:
        """
        Canny edge density as proxy for structural complexity.
        """
        gray  = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = edges.sum() / (255 * edges.size)
        return float(np.clip(density * 10, 0, 1))

    def _saturation_score(self, patch: np.ndarray) -> float:
        """
        High saturation = unusual mineral coloring.
        """
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean() / 255.0
        return float(sat)

    # ── Main scoring ──────────────────────────────────────────────────────────

    def score(self, patch: np.ndarray, label: str,
              confidence: float = 1.0) -> float:
        """
        Compute Si for a given image patch and class label.

        Si = base × (w1·color_anomaly + w2·texture + w3·edge + w4·saturation)
             × detection_confidence
        """
        if patch is None or patch.size == 0:
            return BASE_SCORES.get(label.lower(), 0.3)

        # Ensure minimum patch size
        if patch.shape[0] < 10 or patch.shape[1] < 10:
            return BASE_SCORES.get(label.lower(), 0.3) * confidence

        base   = BASE_SCORES.get(label.lower(), BASE_SCORES["unknown"])

        c_anom = self._color_anomaly(patch)
        tex    = self._texture_richness(patch)
        edge   = self._edge_complexity(patch)
        sat    = self._saturation_score(patch)

        # Weighted combination
        visual = (0.30 * c_anom +
                  0.30 * tex    +
                  0.20 * edge   +
                  0.20 * sat)

        si = float(np.clip(base * (0.6 + 0.4 * visual) * confidence, 0, 1))

        # Track history
        if label not in self._score_history:
            self._score_history[label] = []
        self._score_history[label].append(si)

        log.debug(f"Si({label}) = {si:.4f}  "
                  f"[color={c_anom:.2f} tex={tex:.2f} "
                  f"edge={edge:.2f} sat={sat:.2f}]")
        return si

    def score_from_box(self, frame: np.ndarray,
                       box: Tuple[float, float, float, float],
                       label: str, confidence: float = 1.0) -> float:
        """Crop patch from bounding box and score."""
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        patch = frame[y1:y2, x1:x2]
        return self.score(patch, label, confidence)

    def history_summary(self) -> Dict:
        return {
            k: {"count": len(v), "avg": round(np.mean(v), 4),
                "max": round(max(v), 4)}
            for k, v in self._score_history.items()
        }
