"""Data loaders for POIs, population, and boundaries."""

from typing import Literal

import geopandas as gpd


def pois_from_overture(
    bbox: tuple[float, float, float, float],
    categories: list[str] | None = None,
    release: str = "2024-12-18-0",
) -> gpd.GeoDataFrame:
    """Fetch POIs from Overture Places via DuckDB."""
    raise NotImplementedError


def pois_from_osm(
    bbox: tuple[float, float, float, float], tags: dict | None = None
) -> gpd.GeoDataFrame:
    """Fetch POIs from OpenStreetMap via osmnx."""
    raise NotImplementedError


def pois_from_foursquare(
    bbox: tuple[float, float, float, float], categories: list[str] | None = None
) -> gpd.GeoDataFrame:
    """Fetch POIs from Foursquare Open Source Places."""
    raise NotImplementedError


def population_from_census(
    state: str,
    county: str,
    geography: Literal["tract", "block_group"] = "tract",
    year: int = 2022,
) -> gpd.GeoDataFrame:
    """Load US Census population with demographics."""
    raise NotImplementedError


def boundaries_from_census(
    state: str,
    county: str,
    geography: Literal["tract", "block_group"] = "tract",
    year: int = 2022,
) -> gpd.GeoDataFrame:
    """Load Census geographic boundaries."""
    raise NotImplementedError


def load_fuas(path: str) -> gpd.GeoDataFrame:
    """Load GHSL Functional Urban Areas."""
    raise NotImplementedError


def bbox_from_fua(
    fuas: gpd.GeoDataFrame, fua_name: str, buffer_m: float = 5000
) -> tuple[float, float, float, float]:
    """Get buffered bounding box for a FUA."""
    raise NotImplementedError
