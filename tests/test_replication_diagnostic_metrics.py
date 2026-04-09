from __future__ import annotations

import numpy as np

from pptrain.replication.diagnostics import _jensen_shannon_divergence


def test_jensen_shannon_divergence_is_symmetric_and_finite() -> None:
    left = np.asarray([[1.0, 0.0, -1.0], [0.5, -0.2, 0.3]], dtype=np.float32)
    right = np.asarray([[0.1, 0.2, -0.1], [-0.4, 0.8, 0.0]], dtype=np.float32)

    left_right = _jensen_shannon_divergence(left, right)
    right_left = _jensen_shannon_divergence(right, left)

    assert np.isfinite(left_right)
    assert np.isfinite(right_left)
    assert abs(left_right - right_left) < 1e-9
    assert left_right >= 0.0
