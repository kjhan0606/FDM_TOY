"""Fail-closed vector checks for the wave-only CPU re-audit."""

import pytest

from scripts import audit_wave_only_momentum_null as audit


def test_wave_only_audit_requires_finite_three_vector() -> None:
    assert audit._vector([1.0, 2.0, 3.0], "test").tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="invalid"):
        audit._vector([1.0, 2.0], "short")
    with pytest.raises(ValueError, match="invalid"):
        audit._vector([float("nan"), 0.0, 0.0], "nan")
