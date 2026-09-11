"""Resolve configurable channel types and record channel-selection provenance.

Input: a catalog entry with an optional exclude_channel_types list containing
eeg, mag, or grad. Output: sorted configured types with an explicit empty-list
default. Provenance also records the existing fixed channel-name exclusions
from EXCLUDE_DICT and the shared auxiliary-name exclusions.
"""

from collections.abc import Mapping
from typing import Any

from factory.brain_constant import EXCLUDE_DICT

CHANNEL_SELECTION_KEYS = frozenset({"exclude_channel_types"})
NEURAL_CHANNEL_TYPES = frozenset({"eeg", "mag", "grad"})
AUXILIARY_CHANNEL_NAMES = ("HEO", "VEO", "EKG", "EMG")


def resolve_channel_selection(
    definition: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Validate configured channel-type exclusions.

    Parameters
    ----------
    definition : Mapping[str, Any]
        Catalog entry; exclude_channel_types is optional and defaults to [].

    Returns
    -------
    dict[str, list[str]]
        Sorted excluded types. Invalid or duplicate entries raise ValueError.
    """
    key = "exclude_channel_types"
    values = definition.get(key, [])
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{key} must be a list of nonempty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{key} must not contain duplicates: {values}.")
    unsupported = set(values) - NEURAL_CHANNEL_TYPES
    if unsupported:
        raise ValueError(
            f"Unsupported excluded channel types: {sorted(unsupported)}. "
            f"Expected types from {sorted(NEURAL_CHANNEL_TYPES)}."
        )
    return {key: sorted(values)}


def channel_selection_provenance(
    dataset: str, definition: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Combine fixed names and configured types for filtering and provenance.

    Parameters
    ----------
    dataset : str
        Dataset ID selecting the existing EXCLUDE_DICT entry.
    definition : Mapping[str, Any]
        Catalog entry supplying only configurable channel-type exclusions.

    Returns
    -------
    dict[str, list[str]]
        Actual name/type exclusions; names are never read from the catalog.
    """
    return {
        "exclude_channels": sorted(
            set(EXCLUDE_DICT.get(dataset, [])) | set(AUXILIARY_CHANNEL_NAMES)
        ),
        **resolve_channel_selection(definition),
    }
