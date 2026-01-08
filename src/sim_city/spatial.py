"""City generation and spatial structures.

This module handles the spatial layout of cities, including:
- Synthetic city generation (monocentric, polycentric, etc.)
- Real city loading from data sources
- POI and zone management
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd

    from sim_city.core import SimulationConfig


@dataclass
class Location:
    """A point of interest in the city."""

    id: str
    x: float
    y: float
    zone_id: int
    zone_name: str
    attractiveness: float = 1.0  # For gravity model
    category: str = "general"


@dataclass
class Zone:
    """An areal unit (tract, neighborhood, etc.)."""

    id: int
    name: str
    bounds: tuple[float, float, float, float] | None = None  # xmin, xmax, ymin, ymax


class SpatialEnvironment:
    """
    Manages the spatial layout: POIs, zones, distance matrices.

    Supports both synthetic city generation and (future) real POI loading.
    """

    def __init__(self, config: "SimulationConfig"):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.pois: list[Location] = []
        self.zones: list[Zone] = []
        self.poi_coords: np.ndarray | None = None  # (n_pois, 2)

    def generate_synthetic_city(self, city_type: str = "polycentric") -> pd.DataFrame:
        """
        Generate a synthetic city layout.

        Args:
            city_type: "monocentric", "polycentric", "composite", "urban_villages"

        Returns:
            DataFrame of POIs
        """
        generators = {
            "monocentric": self._make_monocentric,
            "polycentric": self._make_polycentric,
            "composite": self._make_composite,
            "urban_villages": self._make_urban_villages,
        }

        if city_type not in generators:
            raise ValueError(f"Unknown city type: {city_type}")

        # Generators now return (points, zone_ids)
        poi_pts, zone_ids = generators[city_type]()

        # Zone names
        zone_names_map = {0: "Center", 1: "NW", 2: "NE", 3: "SE", 4: "SW"}
        zone_names = np.array([zone_names_map[z] for z in zone_ids])

        # Create Location objects
        self.pois = []
        for i in range(len(poi_pts)):
            loc = Location(
                id=f"poi_{i:04d}",
                x=poi_pts[i, 0],
                y=poi_pts[i, 1],
                zone_id=zone_ids[i],
                zone_name=zone_names[i],
                attractiveness=self.rng.lognormal(0, 0.5),  # Heterogeneous attractiveness
            )
            self.pois.append(loc)

        self.poi_coords = poi_pts

        # Create zones
        self.zones = [
            Zone(0, "Center"),
            Zone(1, "NW"),
            Zone(2, "NE"),
            Zone(3, "SE"),
            Zone(4, "SW"),
        ]

        return self.get_poi_dataframe()

    def _gaussian_mixture(
        self,
        n: int,
        centers: list[tuple[float, float]],
        sigmas: list[float],
        weights: list[float] | None = None,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Sample from 2D Gaussian mixture.

        Args:
            n: Number of points to sample
            centers: List of (x, y) center coordinates
            sigmas: List of standard deviations
            weights: Mixture weights (optional)
            return_components: If True, also return component assignments

        Returns:
            pts: (n, 2) array of coordinates
            components: (n,) array of component indices (if return_components=True)
        """
        if n <= 0:
            if return_components:
                return np.empty((0, 2)), np.empty(0, dtype=int)
            return np.empty((0, 2))

        centers = np.array(centers)
        sigmas = np.array(sigmas)
        k = len(centers)

        if weights is None:
            weights = np.ones(k) / k
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        comp = self.rng.choice(k, size=n, p=weights)
        pts = np.empty((n, 2))

        for j in range(k):
            idx = np.where(comp == j)[0]
            if len(idx) > 0:
                pts[idx, 0] = self.rng.normal(centers[j, 0], sigmas[j], size=len(idx))
                pts[idx, 1] = self.rng.normal(centers[j, 1], sigmas[j], size=len(idx))

        if return_components:
            return pts, comp
        return pts

    def _make_monocentric(self) -> tuple[np.ndarray, np.ndarray]:
        """Single CBD with tight POI clustering. Everyone is 'Center' zone."""
        pts = self._gaussian_mixture(
            self.config.n_pois, centers=[(0.0, 0.0)], sigmas=[1.5]
        )
        # All from single center = all zone 0 (Center)
        zones = np.zeros(len(pts), dtype=int)
        return pts, zones

    def _make_polycentric(self) -> tuple[np.ndarray, np.ndarray]:
        """Multiple centers with slight CBD emphasis."""
        a = self.config.city_extent * 0.6
        # Order: NW, NE, SE, SW, Center
        centers = [(-a, a), (a, a), (a, -a), (-a, -a), (0.0, 0.0)]

        pts, components = self._gaussian_mixture(
            self.config.n_pois,
            centers=centers,
            sigmas=[0.8] * 5,
            weights=[1, 1, 1, 1, 1.5],
            return_components=True,
        )

        # Map components to zone IDs: Center=0, NW=1, NE=2, SE=3, SW=4
        # Component order: 0=NW, 1=NE, 2=SE, 3=SW, 4=Center
        component_to_zone = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
        zones = np.array([component_to_zone[c] for c in components])

        return pts, zones

    def _make_composite(self) -> tuple[np.ndarray, np.ndarray]:
        """Polycentric population but CBD-heavy amenities."""
        a = self.config.city_extent * 0.6
        # Order: NW, NE, SE, SW, Center
        centers = [(-a, a), (a, a), (a, -a), (-a, -a), (0.0, 0.0)]

        pts, components = self._gaussian_mixture(
            self.config.n_pois,
            centers=centers,
            sigmas=[0.8] * 5,
            weights=[1, 1, 1, 1, 6],  # Heavy CBD weight
            return_components=True,
        )

        # Map components to zone IDs: Center=0, NW=1, NE=2, SE=3, SW=4
        component_to_zone = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
        zones = np.array([component_to_zone[c] for c in components])

        return pts, zones

    def _make_urban_villages(self) -> tuple[np.ndarray, np.ndarray]:
        """16-village grid with tight local clustering."""
        xs = [-6, -2, 2, 6]
        ys = [-6, -2, 2, 6]
        centers = [(x, y) for y in ys for x in xs]
        n_villages = len(centers)

        # Map each village center to a zone based on its location
        # Zone mapping: Center=0, NW=1, NE=2, SE=3, SW=4
        village_to_zone = []
        for x, y in centers:
            if abs(x) <= 2 and abs(y) <= 2:
                village_to_zone.append(0)  # Center
            elif x < 0 and y > 0:
                village_to_zone.append(1)  # NW
            elif x > 0 and y > 0:
                village_to_zone.append(2)  # NE
            elif x > 0 and y < 0:
                village_to_zone.append(3)  # SE
            else:  # x < 0 and y < 0
                village_to_zone.append(4)  # SW

        # Distribute POIs across villages, handling remainder
        base_count = self.config.n_pois // n_villages
        remainder = self.config.n_pois % n_villages
        counts = [base_count + (1 if i < remainder else 0) for i in range(n_villages)]

        pts_list = []
        zones_list = []
        for i, center in enumerate(centers):
            if counts[i] > 0:
                pts = self._gaussian_mixture(counts[i], [center], [0.5])
                pts_list.append(pts)
                # All points from this village get the village's zone
                zones_list.append(np.full(counts[i], village_to_zone[i], dtype=int))

        if pts_list:
            return np.vstack(pts_list), np.concatenate(zones_list)
        else:
            return np.empty((0, 2)), np.empty(0, dtype=int)

    def _assign_zones(
        self, x: np.ndarray, y: np.ndarray, mid_box: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign points to 5 zones: Center + 4 quadrants."""
        x = np.asarray(x)
        y = np.asarray(y)

        in_center = (np.abs(x) <= mid_box) & (np.abs(y) <= mid_box)
        zone_id = np.zeros_like(x, dtype=int)

        zone_id[in_center] = 0
        zone_id[(~in_center) & (x < 0) & (y > 0)] = 1  # NW
        zone_id[(~in_center) & (x > 0) & (y > 0)] = 2  # NE
        zone_id[(~in_center) & (x > 0) & (y < 0)] = 3  # SE
        zone_id[(~in_center) & (x < 0) & (y < 0)] = 4  # SW

        names = np.array(["Center", "NW", "NE", "SE", "SW"])
        return zone_id, names[zone_id]

    def get_poi_dataframe(self) -> pd.DataFrame:
        """Return POIs as DataFrame."""
        return pd.DataFrame(
            [
                {
                    "poi_id": p.id,
                    "x": p.x,
                    "y": p.y,
                    "zone_id": p.zone_id,
                    "zone_name": p.zone_name,
                    "attractiveness": p.attractiveness,
                }
                for p in self.pois
            ]
        )

    def load_real_pois(self, filepath: str) -> pd.DataFrame:
        """Load real POIs from file. (Future implementation)"""
        raise NotImplementedError("Real POI loading coming soon")


class SyntheticCity:
    """
    Generate counterfactual urban structures.

    This is a higher-level interface that wraps SpatialEnvironment
    for the API described in the spec.
    """

    def __init__(
        self,
        city_type: Literal[
            "monocentric", "polycentric", "urban_villages", "edge_city", "grid"
        ],
        extent: float = 10.0,  # km half-width
        n_zones: int = 5,
        n_pois: int = 500,
        seed: int | None = None,
    ):
        self.city_type = city_type
        self.extent = extent
        self.n_zones = n_zones
        self.n_pois = n_pois
        self.seed = seed
        self._pois = None
        self._zones = None

    @property
    def pois(self) -> "gpd.GeoDataFrame":
        """POI locations with categories."""
        raise NotImplementedError("Use SpatialEnvironment for now")

    @property
    def zones(self) -> "gpd.GeoDataFrame":
        """Zone polygons with populations."""
        raise NotImplementedError("Use SpatialEnvironment for now")


class RealCity:
    """City from real POI and population data."""

    def __init__(self):
        self._pois = None
        self._zones = None
        self._bbox = None

    @property
    def pois(self) -> "gpd.GeoDataFrame":
        """POI locations with categories."""
        return self._pois

    @property
    def zones(self) -> "gpd.GeoDataFrame":
        """Zone polygons with populations."""
        return self._zones

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        poi_source: Literal["overture", "osm", "foursquare"] = "overture",
        pop_source: Literal["census", "ghspop", "worldpop"] = "census",
        zone_type: Literal["tract", "h3", "custom"] = "tract",
    ):
        """Build city from bounding box."""
        raise NotImplementedError

    @classmethod
    def from_fua(cls, fua_name: str, **kwargs):
        """Build city from Functional Urban Area name."""
        raise NotImplementedError

    def to_synthetic(self, city_type: str) -> SyntheticCity:
        """Convert to synthetic structure for counterfactuals."""
        raise NotImplementedError
