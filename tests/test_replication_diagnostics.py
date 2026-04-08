import numpy as np

from pptrain.replication.diagnostics import _effective_rank


def test_effective_rank_handles_non_finite_inputs() -> None:
    hidden = np.array(
        [
            [1.0, np.nan, 2.0],
            [np.inf, 3.0, -np.inf],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    rank = _effective_rank(hidden)

    assert np.isfinite(rank)
    assert rank >= 0.0
