"""Time-step order and malformed-vector checks for the force comparison."""

import numpy as np
import pytest

from scripts import compare_strang_momentum_candidate as comparison


def test_candidate_momentum_pair_orders_detect_quadratic_refinement() -> None:
    values = [np.array([-0.7 + 0.4 * factor**2, 0.0, 0.0])
              for factor in (1.0, 0.5, 0.25, 0.125)]
    assert comparison._pair_orders(values) == pytest.approx([2.0, 2.0])


def test_candidate_comparison_rejects_unresolved_or_malformed_vectors() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        comparison._pair_orders([np.zeros(3) for _ in range(4)])
    with pytest.raises(ValueError, match="invalid"):
        comparison._vector({"total": [1.0, 2.0]}, "total")
