"""Compatibility accessors for live-wave run metadata."""

from __future__ import annotations


def saved_interval_count(metadata: dict, config: dict) -> int:
    """Return the diagnostic interval count from new or legacy run records."""

    if "save_number" in metadata:
        count = int(metadata["save_number"])
    else:
        try:
            count = int(config["Save Options"]["Number"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                "run metadata and configuration omit the saved interval count"
            ) from error
    if count < 1:
        raise ValueError("saved interval count must be positive")
    return count
