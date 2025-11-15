"""Utility helpers for geospatial calculations."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

EARTH_RADIUS_MILES = 3958.8


def haversine_distance(coord_a: Iterable[float], coord_b: Iterable[float]) -> float:
    """Return the great-circle distance in miles between two coordinates.

    Args:
        coord_a: Tuple containing the latitude and longitude in decimal
            degrees for the starting point.
        coord_b: Tuple containing the latitude and longitude in decimal
            degrees for the destination.

    Returns:
        The distance between the two coordinates expressed in miles.
    """

    lat1, lon1 = coord_a
    lat2, lon2 = coord_b

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_MILES * c


def to_coordinate(metadata) -> Tuple[float, float]:
    """Extract a `(lat, lon)` tuple from a metadata object or mapping."""

    if metadata is None:
        raise ValueError("Metadata is required to compute coordinates")

    if hasattr(metadata, "latitude") and hasattr(metadata, "longitude"):
        return float(metadata.latitude), float(metadata.longitude)

    try:
        return float(metadata["latitude"]), float(metadata["longitude"])
    except (KeyError, TypeError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid metadata payload") from exc
