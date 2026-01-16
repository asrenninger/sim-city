"""Population and agent demographics.

This module defines individual agents and their demographics,
as well as population-level generation and management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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

    # Work location (None if not employed)
    is_employed: bool = False
    work_x: float | None = None
    work_y: float | None = None
    work_zone_id: int | None = None
    work_zone_name: str | None = None
    workplace_id: str | None = None

    # Work share (fraction of trips that are work-anchored, 0 if not employed)
    work_share: float = 0.0

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

    def generate_population(self, city_type: str | None = None) -> pd.DataFrame:
        """
        Generate agent population with homes distributed according to city type.

        Args:
            city_type: If None, uses config.amenity_structure (residents live
                       near amenities). Can override for backward compatibility.

        Returns:
            DataFrame with agent information
        """
        # Default: residents follow amenity distribution (live near amenities)
        city_type = city_type or self.config.amenity_structure

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
            # Assign zones geometrically based on position (not generative component)
            # This partitions into 5 zones: Center + 4 peripheral quadrants
            zones, _ = self.spatial._assign_zones(pts[:, 0], pts[:, 1], mid_box=2.0)
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

    def _assign_employment(self) -> None:
        """
        Assign workplaces to agents via gravity model.

        Uses gravity model: P(work at j) ∝ size_j / dist_ij^β
        Work share sampled from Beta distribution.
        """
        if not self.config.use_work_anchors:
            return

        if not self.spatial.workplaces:
            raise ValueError(
                "Workplaces must be generated before assigning employment. "
                "Call spatial.generate_workplaces() first."
            )

        n_employed = int(len(self.agents) * self.config.employment_rate)
        employed_indices = self.rng.choice(
            len(self.agents), n_employed, replace=False
        )

        # Precompute home-to-workplace distances
        workplace_coords = self.spatial.workplace_coords
        dist_matrix = cdist(self.agent_coords, workplace_coords)

        # Workplace attractiveness (size proxy)
        sizes = np.array([w.attractiveness for w in self.spatial.workplaces])

        # Beta distribution params for work_share
        mean_ws = self.config.work_share_mean
        conc = self.config.work_share_concentration
        alpha_ws = mean_ws * conc
        beta_ws = (1 - mean_ws) * conc

        for i in employed_indices:
            agent = self.agents[i]

            # Gravity model: P(j) ∝ size_j / dist_ij^beta
            distances = np.maximum(dist_matrix[i], 0.1)  # Avoid division by zero
            weights = sizes / (distances ** self.config.work_gravity_beta)
            probs = weights / weights.sum()

            # Sample workplace
            work_idx = self.rng.choice(len(self.spatial.workplaces), p=probs)
            workplace = self.spatial.workplaces[work_idx]

            # Assign work attributes
            agent.is_employed = True
            agent.work_x = workplace.x
            agent.work_y = workplace.y
            agent.work_zone_id = workplace.zone_id
            agent.work_zone_name = workplace.zone_name
            agent.workplace_id = workplace.id
            agent.work_share = float(self.rng.beta(alpha_ws, beta_ws))

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
                    # Work fields
                    "is_employed": a.is_employed,
                    "work_x": a.work_x,
                    "work_y": a.work_y,
                    "work_zone_id": a.work_zone_id,
                    "work_zone_name": a.work_zone_name,
                    "workplace_id": a.workplace_id,
                    "work_share": a.work_share,
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
