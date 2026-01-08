"""Analysis, metrics, and validation.

This module provides tools for computing mobility and isolation metrics,
including:
- Visit matrices and exposure calculations
- Radius of gyration
- Experienced isolation
- Distribution fitting (Zipf, truncated power law)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sim_city.agents import Agent, Population
    from sim_city.core import MobilitySimulation
    from sim_city.spatial import Location, SpatialEnvironment


@dataclass
class SimulationMetrics:
    """Container for all computed metrics."""

    # Per-agent
    n_unique_locations: np.ndarray
    radius_of_gyration: np.ndarray
    entropy: np.ndarray
    experienced_isolation: np.ndarray

    # Population
    mean_isolation: float
    std_isolation: float
    exposure_matrix: np.ndarray

    # Validation
    zipf_exponent: float
    rg_exponent: float


class MetricsCalculator:
    """
    Computes mobility and isolation metrics.
    """

    def __init__(self, population: "Population", spatial: "SpatialEnvironment"):
        self.population = population
        self.spatial = spatial

    def compute_visit_matrix(self) -> np.ndarray:
        """
        Compute visit count matrix (n_agents × n_pois).
        """
        n_agents = len(self.population.agents)
        n_pois = len(self.spatial.pois)

        V = np.zeros((n_agents, n_pois), dtype=int)

        for i, agent in enumerate(self.population.agents):
            for poi_id, count in agent.state.visited_locations.items():
                poi_idx = int(poi_id.split("_")[1])
                V[i, poi_idx] = count

        return V

    def compute_exposure_matrix(
        self, V: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Compute exposure/isolation metrics.

        Args:
            V: Visit matrix (n_agents × n_pois)

        Returns:
            E_df: Zone-to-zone exposure matrix
            person_iso_df: Per-person experienced isolation
            stats: Summary statistics dict
        """
        n_agents, n_pois = V.shape

        # Get zone assignments
        agent_zones = np.array([a.home_zone_id for a in self.population.agents])
        poi_zones = np.array([p.zone_id for p in self.spatial.pois])

        zones = np.unique(agent_zones)
        n_zones = len(zones)
        zone_to_idx = {z: i for i, z in enumerate(zones)}

        # Aggregate visits by zone to each POI
        # group_visits[g, j] = total visits from zone g to POI j
        group_visits = np.zeros((n_zones, n_pois))
        for g in zones:
            agent_mask = agent_zones == g
            group_visits[zone_to_idx[g], :] = V[agent_mask, :].sum(axis=0)

        poi_totals = group_visits.sum(axis=0, keepdims=True)

        # Exposure matrix E[g, h]
        E = np.zeros((n_zones, n_zones))
        g_totals = group_visits.sum(axis=1, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            # S[g,j] = share of g's visits that go to POI j
            S = np.divide(group_visits, np.clip(g_totals, 1e-12, None))
            # H[h,j] = share of POI j's visitors from zone h
            H = np.divide(group_visits, np.clip(poi_totals, 1e-12, None))

        for g in range(n_zones):
            weights = S[g, :]
            if weights.sum() > 0:
                E[g, :] = (weights[np.newaxis, :] * H).sum(axis=1)

        # Per-person experienced isolation
        person_iso = np.zeros(n_agents)

        # Precompute same-zone share at each POI
        same_share_by_zone = {}
        for z in zones:
            idx = zone_to_idx[z]
            same_share_by_zone[z] = np.divide(
                group_visits[idx, :], np.clip(poi_totals.ravel(), 1e-12, None)
            )

        # Person-level visit shares
        person_totals = V.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            visit_share = np.divide(
                V, person_totals[:, np.newaxis], where=(person_totals[:, np.newaxis] > 0)
            )

        for i in range(n_agents):
            zone = agent_zones[i]
            same_share = same_share_by_zone[zone]
            person_iso[i] = (visit_share[i, :] * same_share).sum()

        # Build DataFrames
        zone_names = ["Center", "NW", "NE", "SE", "SW"]
        E_df = pd.DataFrame(
            E, index=zone_names[:n_zones], columns=zone_names[:n_zones]
        )
        E_df = E_df.reset_index(names=["origin_zone"])

        person_iso_df = pd.DataFrame(
            {
                "agent_id": [a.id for a in self.population.agents],
                "zone_id": agent_zones,
                "zone_name": [zone_names[z] for z in agent_zones],
                "experienced_isolation": person_iso,
            }
        )

        # Summary stats
        stats = {
            "total_visits": int(V.sum()),
            "unique_poi_visits": int((V.sum(axis=0) > 0).sum()),
            "mean_places_per_person": float(
                np.mean([(V[i] > 0).sum() for i in range(n_agents)])
            ),
            "mean_visits_per_person": float(V.sum() / n_agents) if n_agents > 0 else 0.0,
            "mean_isolation": float(person_iso.mean()) if len(person_iso) > 0 else 0.0,
            "std_isolation": float(person_iso.std()) if len(person_iso) > 0 else 0.0,
            "min_isolation": float(person_iso.min()) if len(person_iso) > 0 else 0.0,
            "max_isolation": float(person_iso.max()) if len(person_iso) > 0 else 0.0,
        }

        return E_df, person_iso_df, stats

    def compute_rg_distribution(self) -> np.ndarray:
        """Compute radius of gyration for each agent."""
        rg_values = []

        for agent in self.population.agents:
            if not agent.state.visited_locations:
                rg_values.append(0.0)
                continue

            # Get visited location coordinates
            coords = []
            weights = []
            for poi_id, count in agent.state.visited_locations.items():
                poi_idx = int(poi_id.split("_")[1])
                poi = self.spatial.pois[poi_idx]
                coords.append([poi.x, poi.y])
                weights.append(count)

            coords = np.array(coords)
            weights = np.array(weights)

            # Weighted centroid
            total_weight = weights.sum()
            if total_weight == 0:
                rg_values.append(0.0)
                continue

            centroid = (coords * weights[:, np.newaxis]).sum(axis=0) / total_weight

            # Radius of gyration
            squared_distances = ((coords - centroid) ** 2).sum(axis=1)
            rg = np.sqrt((squared_distances * weights).sum() / total_weight)
            rg_values.append(rg)

        return np.array(rg_values)


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================


def compute_metrics(sim: "MobilitySimulation") -> SimulationMetrics:
    """Compute all metrics from simulation results."""
    calc = MetricsCalculator(sim.population, sim.spatial)
    V = calc.compute_visit_matrix()
    _, person_iso_df, stats = calc.compute_exposure_matrix(V)
    rg_values = calc.compute_rg_distribution()

    # Compute entropy for each agent
    entropy_values = []
    for i in range(len(sim.population.agents)):
        visits = V[i, V[i, :] > 0]
        if len(visits) > 0:
            probs = visits / visits.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-12))
        else:
            entropy = 0.0
        entropy_values.append(entropy)

    # Fit distributions (placeholder values for now)
    zipf_exp, _ = fit_zipf(V.sum(axis=0))
    rg_exp, _, _ = fit_truncated_powerlaw(rg_values[rg_values > 0])

    return SimulationMetrics(
        n_unique_locations=np.array(
            [a.state.n_unique_locations for a in sim.population.agents]
        ),
        radius_of_gyration=rg_values,
        entropy=np.array(entropy_values),
        experienced_isolation=person_iso_df["experienced_isolation"].values,
        mean_isolation=stats["mean_isolation"],
        std_isolation=stats["std_isolation"],
        exposure_matrix=sim.exposure_df.set_index("origin_zone").values
        if sim.exposure_df is not None
        else np.array([]),
        zipf_exponent=zipf_exp,
        rg_exponent=rg_exp,
    )


def radius_of_gyration(
    locations: list[tuple[float, float]], weights: list[int] | None = None
) -> float:
    """
    Compute radius of gyration for a set of locations.

    Args:
        locations: List of (x, y) coordinates
        weights: Optional visit counts for weighting

    Returns:
        Radius of gyration value
    """
    if not locations:
        return 0.0

    coords = np.array(locations)
    if weights is None:
        weights = np.ones(len(locations))
    else:
        weights = np.array(weights)

    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0

    # Weighted centroid
    centroid = (coords * weights[:, np.newaxis]).sum(axis=0) / total_weight

    # Radius of gyration
    squared_distances = ((coords - centroid) ** 2).sum(axis=1)
    rg = np.sqrt((squared_distances * weights).sum() / total_weight)

    return float(rg)


def experienced_isolation(
    agent: "Agent",
    pois: list["Location"],
    zone_col: str = "zone_id",
) -> float:
    """
    Compute experienced isolation for an agent.

    Fraction of visits to own-zone POIs.
    """
    if not agent.state.visited_locations:
        return 0.0

    home_zone = agent.home_zone_id
    same_zone_visits = 0
    total_visits = 0

    for poi_id, count in agent.state.visited_locations.items():
        poi_idx = int(poi_id.split("_")[1])
        poi = pois[poi_idx]
        if poi.zone_id == home_zone:
            same_zone_visits += count
        total_visits += count

    if total_visits == 0:
        return 0.0

    return same_zone_visits / total_visits


def exposure_matrix(
    population: "Population",
    pois: list["Location"],
    n_zones: int,
) -> np.ndarray:
    """
    Compute zone-to-zone exposure matrix.

    Args:
        population: Population of agents
        pois: List of POI locations
        n_zones: Number of zones

    Returns:
        n_zones × n_zones exposure matrix
    """
    # Build visit matrix first
    n_agents = len(population.agents)
    n_pois = len(pois)

    V = np.zeros((n_agents, n_pois), dtype=int)
    for i, agent in enumerate(population.agents):
        for poi_id, count in agent.state.visited_locations.items():
            poi_idx = int(poi_id.split("_")[1])
            V[i, poi_idx] = count

    # Get zone assignments
    agent_zones = np.array([a.home_zone_id for a in population.agents])
    poi_zones = np.array([p.zone_id for p in pois])

    # Aggregate visits by zone
    group_visits = np.zeros((n_zones, n_pois))
    for g in range(n_zones):
        agent_mask = agent_zones == g
        group_visits[g, :] = V[agent_mask, :].sum(axis=0)

    poi_totals = group_visits.sum(axis=0, keepdims=True)
    g_totals = group_visits.sum(axis=1, keepdims=True)

    # Compute exposure
    E = np.zeros((n_zones, n_zones))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.divide(group_visits, np.clip(g_totals, 1e-12, None))
        H = np.divide(group_visits, np.clip(poi_totals, 1e-12, None))

    for g in range(n_zones):
        weights = S[g, :]
        if weights.sum() > 0:
            E[g, :] = (weights[np.newaxis, :] * H).sum(axis=1)

    return E


def fit_zipf(visit_counts: np.ndarray) -> tuple[float, float]:
    """
    Fit Zipf exponent to POI popularity.

    Args:
        visit_counts: Array of visit counts per POI

    Returns:
        (exponent, r_squared)
    """
    counts = visit_counts[visit_counts > 0]
    if len(counts) < 2:
        return -1.0, 0.0

    # Sort by frequency (descending)
    sorted_counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)

    # Log-log linear fit
    log_ranks = np.log(ranks)
    log_counts = np.log(sorted_counts)

    # Simple linear regression
    n = len(log_ranks)
    sum_x = log_ranks.sum()
    sum_y = log_counts.sum()
    sum_xy = (log_ranks * log_counts).sum()
    sum_x2 = (log_ranks**2).sum()

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

    # R-squared
    y_mean = sum_y / n
    ss_tot = ((log_counts - y_mean) ** 2).sum()
    intercept = (sum_y - slope * sum_x) / n
    y_pred = intercept + slope * log_ranks
    ss_res = ((log_counts - y_pred) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return float(slope), float(r_squared)


def fit_truncated_powerlaw(
    rg_values: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fit P(Rg) ∝ Rg^(-β) × exp(-Rg/κ).

    Args:
        rg_values: Array of radius of gyration values

    Returns:
        (beta, kappa, r_squared)
    """
    if len(rg_values) < 10:
        return 1.65, 100.0, 0.0  # Return literature defaults

    # For now, return approximate literature values
    # Full MLE fitting would require scipy.optimize
    return 1.65, 100.0, 0.0
