import pytest

from fdm_smbh_delay.run_metadata import saved_interval_count


def test_saved_interval_count_prefers_metadata() -> None:
    assert saved_interval_count(
        {"save_number": 512}, {"Save Options": {"Number": 32}}
    ) == 512


def test_saved_interval_count_supports_legacy_config() -> None:
    assert saved_interval_count({}, {"Save Options": {"Number": 32}}) == 32
    with pytest.raises(ValueError, match="omit"):
        saved_interval_count({}, {})
    with pytest.raises(ValueError, match="positive"):
        saved_interval_count({"save_number": 0}, {})
