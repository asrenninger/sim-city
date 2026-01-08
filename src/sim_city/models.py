"""Mobility mechanisms and destination choice models.

This module implements the core mobility mechanisms from the literature:
- Destination choice models (rank-distance, gravity, intervening opportunities)
- EPR (Exploration-Preferential Return) dynamics
- Recency effects
- Capacity constraints
- Income effects on mobility
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_city.agents import Agent
    from sim_city.core import SimulationConfig
    from sim_city.spatial import Location, SpatialEnvironment

# Income distance multipliers from Moro et al. (2021)
INCOME_DISTANCE_MULTIPLIERS = {
    "Q1_lowest": 0.7,
    "Q2": 0.85,
    "Q3": 1.0,
    "Q4": 1.15,
    "Q5_highest": 1.4,
}


# =============================================================================
# DESTINATION CHOICE MODELS
# =============================================================================


class DestinationChoiceModel:
    """Base class for destination choice models."""

    def __init__(self, config: "SimulationConfig"):
        self.config = config

    def compute_probabilities(
        self,
        agent: "Agent",
        distances: np.ndarray,
        pois: list["Location"],
    ) -> np.ndarray:
        """
        Compute probability distribution over POIs for an agent.

        Args:
            agent: The agent making the choice
            distances: Array of distances from agent home to each POI
            pois: List of POI objects

        Returns:
            Probability array (sums to 1)
        """
        raise NotImplementedError


class RankDistanceModel(DestinationChoiceModel):
    """
    Noulas et al. (2012) rank-based intervening opportunities.

    P(j) ∝ 1 / (rank(j) + 1)^α

    where rank(j) is the number of POIs closer than j.
    """

    def compute_probabilities(
        self,
        agent: "Agent",
        distances: np.ndarray,
        pois: list["Location"],
    ) -> np.ndarray:
        # Compute ranks (0 = closest)
        order = np.argsort(distances)
        ranks = np.argsort(order)

        # Rank-based weights
        alpha = self.config.rank_alpha
        weights = 1.0 / (ranks + 1.0) ** alpha

        return weights / weights.sum()


class GravityModel(DestinationChoiceModel):
    """
    Classic gravity model with distance decay.

    P(j) ∝ A_j / d_j^β

    where A_j is attractiveness and d_j is distance.
    """

    def compute_probabilities(
        self,
        agent: "Agent",
        distances: np.ndarray,
        pois: list["Location"],
    ) -> np.ndarray:
        beta = self.config.gravity_beta

        # Get attractiveness
        attractiveness = np.array([p.attractiveness for p in pois])

        # Avoid division by zero
        safe_dist = np.maximum(distances, 0.1)

        weights = attractiveness / (safe_dist**beta)

        # Income stratification adjustment
        if self.config.use_income_stratification:
            quintile = agent.demographics.income_quintile
            key = (
                f"Q{quintile}"
                if quintile in [2, 3, 4]
                else ("Q1_lowest" if quintile == 1 else "Q5_highest")
            )
            # Higher income = willing to travel further = less distance penalty
            weights = weights ** (1 / INCOME_DISTANCE_MULTIPLIERS.get(key, 1.0))

        return weights / weights.sum()


class InterveningOpportunitiesModel(DestinationChoiceModel):
    """
    Stouffer (1940) intervening opportunities model.

    P(j) ∝ 1 / (1 + s_j)

    where s_j is the number of opportunities (POIs) closer than j.

    This is the individual-level version. For zone-level aggregate flows,
    see Simini et al. (2012) radiation model.
    """

    def compute_probabilities(
        self,
        agent: "Agent",
        distances: np.ndarray,
        pois: list["Location"],
    ) -> np.ndarray:
        n_pois = len(pois)

        # For each POI j, count POIs closer than j (excluding j itself)
        # This is the "intervening opportunities" s_j
        s_j = np.zeros(n_pois)
        for j in range(n_pois):
            s_j[j] = np.sum(distances < distances[j])

        # Stouffer-style decay
        weights = 1.0 / (1.0 + s_j)

        return weights / weights.sum()


def get_destination_model(config: "SimulationConfig") -> DestinationChoiceModel:
    """Factory function for destination choice models."""
    from sim_city.core import DestinationModel

    # Handle string values (for backward compatibility)
    model = config.destination_model
    if isinstance(model, str):
        model_value = model
    else:
        model_value = model.value

    # Map to model classes
    if model_value == "rank_distance":
        return RankDistanceModel(config)
    elif model_value == "gravity":
        return GravityModel(config)
    elif model_value in ["intervening_opportunities", "radiation"]:
        return InterveningOpportunitiesModel(config)
    else:
        raise ValueError(f"Unknown model: {model_value}")


# =============================================================================
# EPR ENGINE
# =============================================================================


class EPREngine:
    """
    Exploration-Preferential Return dynamics.

    At each timestep, agent either:
    - EXPLORES a new location with prob P_new = ρ * S^(-γ)
    - RETURNS to a previously visited location with preferential + recency weighting

    Based on Pappalardo et al. (2015) and Barbosa et al. (2015).
    """

    def __init__(
        self,
        config: "SimulationConfig",
        destination_model: DestinationChoiceModel,
        spatial: "SpatialEnvironment",
    ):
        self.config = config
        self.destination_model = destination_model
        self.spatial = spatial
        self.rng = np.random.default_rng(config.seed)

        # Precompute distance matrix (agents × POIs)
        self._distance_matrix: np.ndarray | None = None

    def set_distance_matrix(
        self, agent_coords: np.ndarray, poi_coords: np.ndarray
    ) -> None:
        """Precompute pairwise distances."""
        # (n_agents, n_pois)
        dx = agent_coords[:, 0, np.newaxis] - poi_coords[np.newaxis, :, 0]
        dy = agent_coords[:, 1, np.newaxis] - poi_coords[np.newaxis, :, 1]
        self._distance_matrix = np.sqrt(dx**2 + dy**2)

    def step(self, agent: "Agent", agent_idx: int, timestep: int) -> str:
        """
        Execute one mobility step for an agent.

        Returns the POI id visited.
        """
        S = agent.state.n_unique_locations

        if S == 0:
            # First visit: must explore
            return self._explore(agent, agent_idx, timestep)

        # Exploration probability
        p_explore = agent.rho * (S ** (-agent.gamma))

        # Capacity constraint: if at capacity, can only return
        if self.config.enforce_capacity and S >= agent.location_capacity:
            p_explore = 0.0

        if self.rng.random() < p_explore:
            return self._explore(agent, agent_idx, timestep)
        else:
            return self._preferential_return(agent, timestep)

    def _explore(self, agent: "Agent", agent_idx: int, timestep: int) -> str:
        """Select a new location to explore."""
        distances = self._distance_matrix[agent_idx]

        # Get base probabilities from destination model
        probs = self.destination_model.compute_probabilities(
            agent, distances, self.spatial.pois
        )

        # Zero out already-visited locations
        for poi_id in agent.state.visited_locations:
            poi_idx = int(poi_id.split("_")[1])
            probs[poi_idx] = 0.0

        # Renormalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # All locations visited; force return instead
            return self._preferential_return(agent, timestep)

        # Sample new location
        poi_idx = self.rng.choice(len(self.spatial.pois), p=probs)
        poi_id = self.spatial.pois[poi_idx].id

        agent.record_visit(poi_id, timestep)
        return poi_id

    def _preferential_return(self, agent: "Agent", timestep: int) -> str:
        """Return to a previously visited location with preferential + recency weighting."""
        visited = agent.state.visited_locations

        if not visited:
            raise ValueError("Cannot return with no visit history")

        poi_ids = list(visited.keys())

        # Frequency-based weights (Zipf)
        freq_weights = np.array(
            [visited[pid] ** self.config.return_eta for pid in poi_ids]
        )

        # Recency weights (Barbosa)
        if self.config.use_recency:
            recency_weights = np.zeros(len(poi_ids))
            recent = agent.state.get_recent_locations(self.config.recency_window)

            for i, pid in enumerate(poi_ids):
                if pid in recent:
                    # Position in recency window (0 = most recent)
                    pos = len(recent) - 1 - recent[::-1].index(pid)
                    recency_weights[i] = np.exp(-self.config.recency_decay * pos)
                else:
                    recency_weights[i] = 0.1  # Base weight for non-recent

            weights = freq_weights * recency_weights
        else:
            weights = freq_weights

        # Normalize and sample
        probs = weights / weights.sum()
        chosen_idx = self.rng.choice(len(poi_ids), p=probs)
        poi_id = poi_ids[chosen_idx]

        agent.record_visit(poi_id, timestep)
        return poi_id


# =============================================================================
# STANDALONE FUNCTIONS (for API compatibility)
# =============================================================================


def epr_decision(
    agent: "Agent",
    rho: float = 0.6,
    gamma: float = 0.21,
    rng: np.random.Generator | None = None,
) -> bool:
    """
    Decide explore (True) vs return (False).

    Based on González et al. (2008), Song et al. (2010).
    P(explore) = rho * S^(-gamma) where S = unique locations visited.
    """
    S = agent.state.n_unique_locations
    if S == 0:
        return True
    p_explore = rho * (S**-gamma)
    if rng is None:
        rng = np.random.default_rng()
    return rng.random() < p_explore


def apply_income_effect(
    base_alpha: float, income_quintile: int, effect_size: float = 0.1
) -> float:
    """
    Adjust rank-distance alpha by income.

    Higher income -> lower alpha -> more distant trips.
    Based on Moro et al. (2021).
    """
    return base_alpha - effect_size * (income_quintile - 3)


def destination_choice(
    agent: "Agent",
    pois: list["Location"],
    distances: np.ndarray,
    method: str = "rank_distance",
    alpha: float = 0.84,
    rng: np.random.Generator | None = None,
) -> int:
    """
    Select destination POI for exploration trip.

    Args:
        agent: Agent making the choice
        pois: List of POI locations
        distances: Array of distances from agent to each POI
        method: 'rank_distance', 'gravity', or 'intervening_opportunities'
        alpha: Model parameter
        rng: Random number generator

    Returns:
        Index of selected POI
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == "rank_distance":
        # Compute ranks (0 = closest)
        order = np.argsort(distances)
        ranks = np.argsort(order)
        weights = 1.0 / (ranks + 1.0) ** alpha
    elif method == "gravity":
        attractiveness = np.array([p.attractiveness for p in pois])
        safe_dist = np.maximum(distances, 0.1)
        weights = attractiveness / (safe_dist**alpha)
    elif method == "intervening_opportunities":
        n_pois = len(pois)
        s_j = np.zeros(n_pois)
        for j in range(n_pois):
            s_j[j] = np.sum(distances < distances[j])
        weights = 1.0 / (1.0 + s_j)
    else:
        raise ValueError(f"Unknown method: {method}")

    probs = weights / weights.sum()
    return rng.choice(len(pois), p=probs)


def select_return_location(
    agent: "Agent",
    use_recency: bool = True,
    recency_window: int = 5,
    recency_decay: float = 0.5,
    return_eta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> str:
    """
    Select location for return trip (frequency + recency).

    Args:
        agent: Agent making the return
        use_recency: Whether to apply Barbosa recency weighting
        recency_window: Size of working memory
        recency_decay: Exponential decay rate
        return_eta: Preferential return exponent
        rng: Random number generator

    Returns:
        POI ID of selected location
    """
    if rng is None:
        rng = np.random.default_rng()

    visited = agent.state.visited_locations
    if not visited:
        raise ValueError("Cannot return with no visit history")

    poi_ids = list(visited.keys())

    # Frequency-based weights
    freq_weights = np.array([visited[pid] ** return_eta for pid in poi_ids])

    # Recency weights
    if use_recency:
        recency_weights = np.zeros(len(poi_ids))
        recent = agent.state.get_recent_locations(recency_window)

        for i, pid in enumerate(poi_ids):
            if pid in recent:
                pos = len(recent) - 1 - recent[::-1].index(pid)
                recency_weights[i] = np.exp(-recency_decay * pos)
            else:
                recency_weights[i] = 0.1

        weights = freq_weights * recency_weights
    else:
        weights = freq_weights

    probs = weights / weights.sum()
    chosen_idx = rng.choice(len(poi_ids), p=probs)
    return poi_ids[chosen_idx]


def enforce_capacity(agent: "Agent", new_location: str, capacity: int) -> None:
    """
    Drop oldest location if over capacity.

    Based on Alessandretti et al. (2018) conserved quantity.
    """
    if agent.state.n_unique_locations > capacity:
        # Find oldest location (first visited)
        if agent.state.visit_history:
            # Get unique locations in order of first visit
            seen = set()
            oldest = None
            for poi_id, _ in agent.state.visit_history:
                if poi_id not in seen:
                    if oldest is None:
                        oldest = poi_id
                    seen.add(poi_id)

            if oldest and oldest != new_location:
                del agent.state.visited_locations[oldest]
