"""Population and agent demographics.

This module defines individual agents and their demographics,
as well as population-level generation and management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sim_city.core import SimulationConfig
    from sim_city.spatial import SpatialEnvironment


@dataclass
class Demographics:
    """
    Demographic attributes for heterogeneity modeling.

    Based on Gauvin et al. (2020) and Moro et al. (2021).
    """

    income_quintile: int = 3  # 1-5
    gender: str = "unspecified"
    age_group: str = "adult"


@dataclass
class AgentState:
    """Mutable state tracking for an agent."""

    visited_locations: dict[str, int] = field(
        default_factory=dict
    )  # poi_id -> visit_count
    visit_history: list[tuple[str, int]] = field(
        default_factory=list
    )  # (poi_id, timestep)
    total_visits: int = 0

    @property
    def n_unique_locations(self) -> int:
        return len(self.visited_locations)

    def get_recent_locations(self, window: int = 5) -> list[str]:
        """Get the most recently visited locations."""
        if not self.visit_history:
            return []
        return [loc for loc, _ in self.visit_history[-window:]]


@dataclass
class Agent:
    """
    An individual in the simulation.

    Combines fixed attributes (home, demographics, mobility type) with
    mutable state (visit history, accumulated visits).

    Attributes:
        id: Unique identifier
        home_x: Home x-coordinate
        home_y: Home y-coordinate
        home_zone_id: Zone ID of home location
        home_zone_name: Zone name of home location
        demographics: Demographic attributes for heterogeneity
        mobility_type: Pappalardo returner/explorer dichotomy
        k_rg: Individual Rg characteristic
        rho: Individual EPR exploration coefficient
        gamma: Individual EPR decay exponent
        location_capacity: Alessandretti individual capacity
        state: Mutable visit state
    """

    id: str
    home_x: float
    home_y: float
    home_zone_id: int
    home_zone_name: str

    # Demographics for heterogeneity
    demographics: Demographics = field(default_factory=Demographics)

    # Mobility type (Pappalardo returner/explorer)
    mobility_type: str = "returner"
    k_rg: float = 0.3  # Individual Rg characteristic

    # Individual EPR parameters (can vary)
    rho: float = 0.6
    gamma: float = 0.21

    # Individual location capacity (Alessandretti - varies across population)
    location_capacity: int = 25

    # State
    state: AgentState = field(default_factory=AgentState)

    def record_visit(self, poi_id: str, timestep: int):
        """Record a visit to a location."""
        self.state.visited_locations[poi_id] = (
            self.state.visited_locations.get(poi_id, 0) + 1
        )
        self.state.visit_history.append((poi_id, timestep))
        self.state.total_visits += 1


class Population:
    """
    Manages the agent population.

    Handles generation with appropriate heterogeneity based on config.
    """

    def __init__(self, config: "SimulationConfig", spatial: "SpatialEnvironment"):
        self.config = config
        self.spatial = spatial
        self.rng = np.random.default_rng(config.seed)

        self.agents: list[Agent] = []
        self.agent_coords: np.ndarray | None = None

    def __len__(self):
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents)

    def generate_population(self, city_type: str = "polycentric") -> pd.DataFrame:
        """
        Generate agent population with homes distributed according to city type.

        Args:
            city_type: One of 'monocentric', 'polycentric', 'composite', 'urban_villages'

        Returns:
            DataFrame with agent information
        """
        # Generate home locations with zone assignments from generative process
        home_pts, zone_ids = self._generate_homes(city_type)

        # Zone names
        zone_names_map = {0: "Center", 1: "NW", 2: "NE", 3: "SE", 4: "SW"}
        zone_names = np.array([zone_names_map[z] for z in zone_ids])

        self.agents = []
        for i in range(len(home_pts)):
            # Determine mobility type
            if self.config.use_returner_explorer:
                is_returner = self.rng.random() < self.config.returner_fraction
                mobility_type = "returner" if is_returner else "explorer"
                k_rg = 0.18 if is_returner else 0.78  # Pappalardo values
            else:
                mobility_type = "neutral"
                k_rg = 0.5

            # Demographics
            demographics = Demographics(
                income_quintile=(
                    self.rng.integers(1, 6)
                    if self.config.use_income_stratification
                    else 3
                ),
                gender=(
                    self.rng.choice(["male", "female"])
                    if self.config.use_gender_effects
                    else "unspecified"
                ),
            )

            # Individual EPR variation (±10%)
            rho = self.config.epr_rho * self.rng.uniform(0.9, 1.1)
            gamma = self.config.epr_gamma * self.rng.uniform(0.9, 1.1)

            # Individual location capacity (Alessandretti: ~25 mean, individual variation)
            location_capacity = max(
                10,
                int(
                    self.rng.normal(
                        self.config.location_capacity_mean,
                        self.config.location_capacity_std,
                    )
                ),
            )

            agent = Agent(
                id=f"agent_{i:05d}",
                home_x=home_pts[i, 0],
                home_y=home_pts[i, 1],
                home_zone_id=zone_ids[i],
                home_zone_name=zone_names[i],
                demographics=demographics,
                mobility_type=mobility_type,
                k_rg=k_rg,
                rho=rho,
                gamma=gamma,
                location_capacity=location_capacity,
            )
            self.agents.append(agent)

        self.agent_coords = home_pts
        return self.get_agent_dataframe()

    def _generate_homes(self, city_type: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate home locations (wider spread than POIs).

        Returns:
            pts: (n, 2) array of home coordinates
            zone_ids: (n,) array of zone assignments based on generative component
        """
        n = self.config.n_residents

        if city_type == "monocentric":
            pts = self.spatial._gaussian_mixture(n, centers=[(0.0, 0.0)], sigmas=[4.5])
            # All from single center = all zone 0 (Center)
            zones = np.zeros(n, dtype=int)
            return pts, zones

        elif city_type in ["polycentric", "composite"]:
            a = self.config.city_extent * 0.6
            # Order: NW, NE, SE, SW, Center
            centers = [(-a, a), (a, a), (a, -a), (-a, -a), (0.0, 0.0)]

            pts, components = self.spatial._gaussian_mixture(
                n,
                centers=centers,
                sigmas=[2.2] * 5,
                weights=[1, 1, 1, 1, 1.1],
                return_components=True,
            )

            # Map components to zone IDs: Center=0, NW=1, NE=2, SE=3, SW=4
            component_to_zone = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
            zones = np.array([component_to_zone[c] for c in components])

            return pts, zones

        elif city_type == "urban_villages":
            xs = [-6, -2, 2, 6]
            ys = [-6, -2, 2, 6]
            centers = [(x, y) for y in ys for x in xs]
            n_villages = len(centers)

            # Map each village center to a zone
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
                else:
                    village_to_zone.append(4)  # SW

            # Distribute residents across villages, handling remainder
            base_count = n // n_villages
            remainder = n % n_villages
            counts = [
                base_count + (1 if i < remainder else 0) for i in range(n_villages)
            ]

            pts_list = []
            zones_list = []
            for i, center in enumerate(centers):
                if counts[i] > 0:
                    pts = self.spatial._gaussian_mixture(counts[i], [center], [0.8])
                    pts_list.append(pts)
                    zones_list.append(np.full(counts[i], village_to_zone[i], dtype=int))

            if pts_list:
                return np.vstack(pts_list), np.concatenate(zones_list)
            else:
                return np.empty((0, 2)), np.empty(0, dtype=int)
        else:
            raise ValueError(f"Unknown city type: {city_type}")

    def get_agent_dataframe(self) -> pd.DataFrame:
        """Return agents as DataFrame."""
        return pd.DataFrame(
            [
                {
                    "agent_id": a.id,
                    "home_x": a.home_x,
                    "home_y": a.home_y,
                    "zone_id": a.home_zone_id,
                    "zone_name": a.home_zone_name,
                    "mobility_type": a.mobility_type,
                    "k_rg": a.k_rg,
                    "income_quintile": a.demographics.income_quintile,
                    "gender": a.demographics.gender,
                }
                for a in self.agents
            ]
        )

    @classmethod
    def from_zones(cls, zones, config, rng):
        """Generate agents distributed across zones."""
        raise NotImplementedError

    @classmethod
    def from_census(cls, census_data, config, rng):
        """Generate agents from real demographic data."""
        raise NotImplementedError
