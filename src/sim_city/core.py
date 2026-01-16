"""Configuration and simulation engine.

This module contains the main orchestration classes for running
mobility simulations, including configuration management and the
primary simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sim_city.agents import Population
    from sim_city.metrics import MetricsCalculator, SimulationMetrics
    from sim_city.models import DestinationChoiceModel, EPREngine
    from sim_city.spatial import SpatialEnvironment, SyntheticCity


class DestinationModel(Enum):
    """Available destination choice models."""

    RANK_DISTANCE = "rank_distance"  # Noulas et al. (2012)
    GRAVITY = "gravity"  # Classic gravity model
    INTERVENING_OPPORTUNITIES = "intervening_opportunities"  # Stouffer (1940)


@dataclass
class SimulationConfig:
    """
    Master configuration for simulation runs.

    Toggle different literature mechanisms on/off to compare their effects.
    All parameters have literature-derived defaults with citations.

    Attributes:
        destination_model: Which destination choice model to use
        rank_alpha: Noulas rank-distance exponent (α=0.84)
        gravity_beta: Gravity model distance decay (β≈2.0)
        use_epr: Enable exploration-preferential return dynamics
        epr_rho: EPR exploration coefficient (ρ=0.6)
        epr_gamma: EPR exploration decay exponent (γ=0.21)
        return_eta: Preferential return exponent (η=1.0)
        use_recency: Enable Barbosa recency weighting
        recency_window: Working memory size (~5 locations)
        recency_decay: Exponential decay rate for recency
        location_capacity_mean: Alessandretti capacity (~25)
        location_capacity_std: Individual variation in capacity
        enforce_capacity: Hard constraint on location count
        use_capacity_decay: Alternative: decay-based capacity
        capacity_decay_window: Timesteps before location becomes inactive
        monthly_visits: Baseline monthly trips
        zipf_zeta: Visit frequency distribution exponent
        use_schlaepfer_scaling: Enable city-size scaling
        schlaepfer_delta: Per-capita interaction scaling exponent
        base_population: Reference population for scaling
        use_returner_explorer: Enable Pappalardo dichotomy
        returner_fraction: Fraction of returners (~75%)
        use_income_stratification: Enable income effects on mobility
        use_gender_effects: Enable gender effects on Rg
        n_timesteps: Number of simulation steps
        burnin_steps: Steps before measuring equilibrium
        convergence_threshold: Threshold for equilibrium detection
        n_residents: Number of agents
        n_pois: Number of points of interest
        city_extent: Half-width of synthetic city (km)
        seed: Random seed for reproducibility
    """

    # --- Destination Choice ---
    destination_model: DestinationModel | str = DestinationModel.RANK_DISTANCE
    rank_alpha: float = 0.84  # Noulas et al. (2012)
    gravity_beta: float = 2.0  # Schläpfer/Pappalardo

    # --- EPR Dynamics ---
    use_epr: bool = True  # Enable exploration-preferential return
    epr_rho: float = 0.6  # Exploration coefficient
    epr_gamma: float = 0.21  # Exploration decay
    return_eta: float = 1.0  # Preferential return exponent

    # --- Recency Effects ---
    use_recency: bool = True  # Barbosa et al. recency weighting
    recency_window: int = 5  # Working memory size
    recency_decay: float = 0.5  # Decay rate

    # --- Capacity Constraints (Alessandretti et al. 2018) ---
    location_capacity_mean: float = 25.0
    location_capacity_std: float = 5.0  # Individual variation
    enforce_capacity: bool = True  # Hard constraint

    # Alternative: decay-based capacity
    use_capacity_decay: bool = False
    capacity_decay_window: int = 30  # Timesteps before location inactive

    monthly_visits: int = 40  # Baseline visits
    zipf_zeta: float = -1.0  # Visit frequency distribution

    # --- City Scaling ---
    use_schlaepfer_scaling: bool = False
    schlaepfer_delta: float = 0.12
    base_population: int = 100_000

    # --- Heterogeneity ---
    use_returner_explorer: bool = False
    returner_fraction: float = 0.75
    use_income_stratification: bool = False
    use_gender_effects: bool = False

    # --- Spatial Structure ---
    # Decouple amenity vs workplace spatial patterns for counterfactual scenarios
    # e.g., "15-min city for daily life but CBD for work"
    amenity_structure: str = "polycentric"  # monocentric, polycentric, urban_villages
    workplace_structure: str = "polycentric"  # monocentric, polycentric, urban_villages

    # Concentration parameters (CBD weight in Gaussian mixture)
    # Higher = more concentration in central business district
    amenity_cbd_weight: float = 1.5  # Default: slight CBD emphasis
    workplace_cbd_weight: float = 4.0  # Default: jobs more concentrated than amenities

    # Clustering tightness (Gaussian sigma)
    amenity_sigma: float = 0.8  # Spread of amenity clusters
    workplace_sigma: float = 0.5  # Tighter clustering for workplaces

    # --- Work Anchors ---
    use_work_anchors: bool = False  # Master toggle for home+work dual anchors
    n_workplaces: int = 100  # Number of workplace locations

    # Employment
    employment_rate: float = 0.67  # Fraction of agents who are employed

    # Work assignment (gravity model)
    work_gravity_beta: float = 1.5  # Distance decay for commute assignment

    # Work share distribution (Beta distribution for trip allocation)
    work_share_mean: float = 0.36  # Mean fraction of trips that are work-anchored
    work_share_concentration: float = 5.0  # Beta concentration parameter

    # --- Simulation Parameters ---
    n_timesteps: int = 100
    burnin_steps: int = 20
    convergence_threshold: float = 0.01

    # --- Scale ---
    n_residents: int = 5000
    n_pois: int = 300
    city_extent: float = 10.0  # Half-width of synthetic city

    # --- Random Seed ---
    seed: int | None = 42

    def describe(self) -> str:
        """Return human-readable description of active mechanisms."""
        active = []

        # Spatial structure
        if self.amenity_structure == self.workplace_structure:
            active.append(f"Structure: {self.amenity_structure}")
        else:
            active.append(
                f"Structure: amenities={self.amenity_structure}, work={self.workplace_structure}"
            )

        # Handle both enum value and string
        if isinstance(self.destination_model, DestinationModel):
            model_name = self.destination_model.value
        else:
            model_name = str(self.destination_model)
        active.append(f"Destination: {model_name}")

        if self.use_epr:
            active.append(f"EPR(ρ={self.epr_rho}, γ={self.epr_gamma})")
        if self.use_recency:
            active.append(f"Recency(window={self.recency_window})")
        if self.enforce_capacity:
            active.append(
                f"Capacity(L={self.location_capacity_mean}±{self.location_capacity_std})"
            )
        if self.use_schlaepfer_scaling:
            active.append(f"CityScaling(δ={self.schlaepfer_delta})")
        if self.use_returner_explorer:
            active.append(f"RetExp(frac={self.returner_fraction})")
        if self.use_work_anchors:
            active.append(
                f"WorkAnchors(emp={self.employment_rate:.0%}, ws={self.work_share_mean:.2f})"
            )
        return " | ".join(active)


@dataclass
class SimulationSnapshot:
    """A snapshot of simulation state at a timestep."""

    timestep: int
    mean_isolation: float
    std_isolation: float
    mean_unique_locations: float
    std_unique_locations: float
    mean_rg: float
    std_rg: float
    zone_isolation: dict[str, float]


class EvolutionTracker:
    """
    Tracks simulation evolution over time.

    Monitors convergence toward equilibrium (like Schelling segregation dynamics).
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.snapshots: list[SimulationSnapshot] = []

    def record_snapshot(
        self,
        timestep: int,
        metrics: "MetricsCalculator",
        person_iso_df: pd.DataFrame,
    ) -> SimulationSnapshot:
        """Record current state."""
        # Compute Rg distribution
        rg_values = metrics.compute_rg_distribution()

        # Compute unique locations distribution
        unique_locs = np.array(
            [a.state.n_unique_locations for a in metrics.population.agents]
        )

        # Zone-level isolation
        zone_iso = (
            person_iso_df.groupby("zone_name")["experienced_isolation"].mean().to_dict()
        )

        snapshot = SimulationSnapshot(
            timestep=timestep,
            mean_isolation=person_iso_df["experienced_isolation"].mean(),
            std_isolation=person_iso_df["experienced_isolation"].std(),
            mean_unique_locations=unique_locs.mean(),
            std_unique_locations=unique_locs.std(),
            mean_rg=rg_values.mean() if len(rg_values) > 0 else 0.0,
            std_rg=rg_values.std() if len(rg_values) > 0 else 0.0,
            zone_isolation=zone_iso,
        )

        self.snapshots.append(snapshot)
        return snapshot

    def check_convergence(self, window: int = 10) -> bool:
        """
        Check if simulation has converged to equilibrium.

        Uses rolling window variance in mean isolation.
        """
        if len(self.snapshots) < window + self.config.burnin_steps:
            return False

        recent = self.snapshots[-window:]
        isolations = [s.mean_isolation for s in recent]
        variance = np.var(isolations)

        return variance < self.config.convergence_threshold**2

    def get_evolution_dataframe(self) -> pd.DataFrame:
        """Return evolution as DataFrame for plotting."""
        records = []
        for s in self.snapshots:
            record = {
                "timestep": s.timestep,
                "mean_isolation": s.mean_isolation,
                "std_isolation": s.std_isolation,
                "mean_unique_locations": s.mean_unique_locations,
                "std_unique_locations": s.std_unique_locations,
                "mean_rg": s.mean_rg,
                "std_rg": s.std_rg,
            }
            for zone, iso in s.zone_isolation.items():
                record[f"isolation_{zone}"] = iso
            records.append(record)

        return pd.DataFrame(records)


class MobilitySimulation:
    """
    Main simulation orchestrator.

    Brings together all components for a complete simulation run.

    Example:
        >>> config = SimulationConfig(n_residents=5000, n_timesteps=100)
        >>> sim = MobilitySimulation(config)
        >>> sim.setup("polycentric")
        >>> sim.run()
        >>> print(f"Mean isolation: {sim.isolation_df['experienced_isolation'].mean():.3f}")
    """

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Components (initialized in setup())
        self.spatial: SpatialEnvironment | None = None
        self.population: Population | None = None
        self.destination_model: DestinationChoiceModel | None = None
        self.epr_engine: EPREngine | None = None
        self.metrics: MetricsCalculator | None = None
        self.tracker: EvolutionTracker | None = None

        # Results
        self.poi_df: pd.DataFrame | None = None
        self.agent_df: pd.DataFrame | None = None
        self.evolution_df: pd.DataFrame | None = None
        self.exposure_df: pd.DataFrame | None = None
        self.isolation_df: pd.DataFrame | None = None

    def setup(self, city_type: str | None = None) -> "MobilitySimulation":
        """
        Initialize all simulation components.

        Args:
            city_type: DEPRECATED. Use config.amenity_structure and config.workplace_structure.
                       If provided, overrides both amenity and workplace structure for
                       backward compatibility.

        Returns:
            self for method chaining
        """
        # Import here to avoid circular imports
        from sim_city.agents import Population
        from sim_city.metrics import MetricsCalculator
        from sim_city.models import EPREngine, get_destination_model
        from sim_city.spatial import SpatialEnvironment

        print(f"Setting up simulation: {self.config.describe()}")

        # Spatial environment
        self.spatial = SpatialEnvironment(self.config)

        # Generate amenities (POIs) - use city_type override if provided for backward compat
        self.poi_df = self.spatial.generate_synthetic_city(city_type=city_type)
        print(f"  Generated {len(self.spatial.pois)} POIs ({self.config.amenity_structure})")

        # Generate workplaces if using work anchors
        self.workplace_df: pd.DataFrame | None = None
        if self.config.use_work_anchors:
            self.workplace_df = self.spatial.generate_workplaces(city_type=city_type)
            print(f"  Generated {len(self.spatial.workplaces)} workplaces ({self.config.workplace_structure})")

        # Population (follows amenity structure by default)
        self.population = Population(self.config, self.spatial)
        self.agent_df = self.population.generate_population(city_type=city_type)
        print(f"  Generated {len(self.population.agents)} agents")

        # Assign employment if using work anchors
        if self.config.use_work_anchors:
            self.population._assign_employment()
            n_employed = sum(a.is_employed for a in self.population.agents)
            print(f"  Employment: {n_employed}/{len(self.population.agents)} ({n_employed/len(self.population.agents):.1%})")
            # Update agent dataframe with employment info
            self.agent_df = self.population.get_agent_dataframe()

        # Destination model
        self.destination_model = get_destination_model(self.config)

        # Get model name for display
        if isinstance(self.config.destination_model, DestinationModel):
            model_name = self.config.destination_model.value
        else:
            model_name = str(self.config.destination_model)
        print(f"  Using {model_name} destination model")

        # EPR engine
        self.epr_engine = EPREngine(self.config, self.destination_model, self.spatial)

        # Set up distance matrices
        if self.config.use_work_anchors:
            # Dual distance matrices for home and work anchors
            work_coords = np.array([
                [a.work_x, a.work_y] if a.is_employed else [np.nan, np.nan]
                for a in self.population.agents
            ])
            employed_mask = np.array([a.is_employed for a in self.population.agents])
            self.epr_engine.set_dual_distance_matrices(
                self.population.agent_coords,
                work_coords,
                self.spatial.poi_coords,
                employed_mask,
            )
        else:
            # Standard home-only distance matrix
            self.epr_engine.set_distance_matrix(
                self.population.agent_coords, self.spatial.poi_coords
            )

        # Metrics calculator
        self.metrics = MetricsCalculator(self.population, self.spatial)

        # Evolution tracker
        self.tracker = EvolutionTracker(self.config)

        return self

    def load_city(self, city: "SyntheticCity") -> "MobilitySimulation":
        """
        Load a pre-built city (alternative to setup()).

        Args:
            city: A SyntheticCity or RealCity instance

        Returns:
            self for method chaining
        """
        raise NotImplementedError("Use setup() for now; load_city() coming soon")

    def run(
        self, verbose: bool = True, snapshot_interval: int = 5
    ) -> "MobilitySimulation":
        """
        Run the simulation for n_timesteps.

        At each timestep, each agent makes one mobility decision.

        Args:
            verbose: Print progress updates
            snapshot_interval: How often to record snapshots

        Returns:
            self for method chaining
        """
        if self.spatial is None:
            raise ValueError("Must call setup() before run()")

        print(f"\nRunning simulation for {self.config.n_timesteps} timesteps...")

        # City scaling adjustment
        if self.config.use_schlaepfer_scaling:
            scale = (
                self.config.n_residents / self.config.base_population
            ) ** self.config.schlaepfer_delta
            visits_per_step = int(np.round(1 * scale))
        else:
            visits_per_step = 1

        for t in range(self.config.n_timesteps):
            # Each agent takes one step
            for i, agent in enumerate(self.population.agents):
                for _ in range(visits_per_step):
                    if self.config.use_epr:
                        self.epr_engine.step(agent, i, t)
                    else:
                        # Simple destination choice without EPR
                        self._simple_visit(agent, i, t)

            # Record snapshot
            if t % snapshot_interval == 0 or t == self.config.n_timesteps - 1:
                V = self.metrics.compute_visit_matrix()
                _, person_iso_df, _ = self.metrics.compute_exposure_matrix(V)
                snapshot = self.tracker.record_snapshot(t, self.metrics, person_iso_df)

                if verbose:
                    print(
                        f"  t={t:4d}: isolation={snapshot.mean_isolation:.3f} "
                        f"± {snapshot.std_isolation:.3f}, "
                        f"locs={snapshot.mean_unique_locations:.1f}, "
                        f"Rg={snapshot.mean_rg:.2f}"
                    )

                # Check convergence
                if self.tracker.check_convergence():
                    print(f"  Converged at t={t}")
                    break

        # Final metrics
        V = self.metrics.compute_visit_matrix()
        self.exposure_df, self.isolation_df, final_stats = (
            self.metrics.compute_exposure_matrix(V)
        )
        self.evolution_df = self.tracker.get_evolution_dataframe()

        print("\nFinal statistics:")
        for k, v in final_stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

        return self

    def _simple_visit(self, agent, agent_idx: int, timestep: int):
        """Non-EPR visit (for comparison)."""
        distances = self.epr_engine._distance_matrix[agent_idx]
        probs = self.destination_model.compute_probabilities(
            agent, distances, self.spatial.pois
        )

        # Apply capacity constraint
        if self.config.enforce_capacity:
            if agent.state.n_unique_locations >= agent.location_capacity:
                # Must return to existing
                for poi_id in agent.state.visited_locations:
                    probs[int(poi_id.split("_")[1])] *= 10  # Boost visited

        probs = probs / probs.sum()
        poi_idx = self.rng.choice(len(self.spatial.pois), p=probs)
        poi_id = self.spatial.pois[poi_idx].id
        agent.record_visit(poi_id, timestep)

    def plot_diagnostics(self, save_path: str | None = None):
        """
        Generate 12-panel diagnostic dashboard.

        Args:
            save_path: Optional path to save the figure
        """
        from sim_city.viz import DiagnosticsSuite

        diag = DiagnosticsSuite(self)
        return diag.plot_all(save_path=save_path)

    def save_results(self, output_dir: str) -> dict[str, str]:
        """
        Save all results to CSV files.

        Args:
            output_dir: Directory to save files

        Returns:
            Dict mapping result names to file paths
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        files = {}

        # POIs
        self.poi_df.to_csv(out / "pois.csv", index=False)
        files["pois"] = str(out / "pois.csv")

        # Agents (with final state)
        agent_final = self.agent_df.copy()
        agent_final["n_locations"] = [
            a.state.n_unique_locations for a in self.population.agents
        ]
        agent_final["total_visits"] = [
            a.state.total_visits for a in self.population.agents
        ]
        agent_final.to_csv(out / "agents.csv", index=False)
        files["agents"] = str(out / "agents.csv")

        # Evolution
        self.evolution_df.to_csv(out / "evolution.csv", index=False)
        files["evolution"] = str(out / "evolution.csv")

        # Exposure matrix
        self.exposure_df.to_csv(out / "exposure_matrix.csv", index=False)
        files["exposure_matrix"] = str(out / "exposure_matrix.csv")

        # Per-person isolation
        self.isolation_df.to_csv(out / "isolation.csv", index=False)
        files["isolation"] = str(out / "isolation.csv")

        # Visit matrix (sparse format)
        V = self.metrics.compute_visit_matrix()
        rows, cols = np.nonzero(V)
        visits_df = pd.DataFrame(
            {
                "agent_id": [self.population.agents[r].id for r in rows],
                "poi_id": [self.spatial.pois[c].id for c in cols],
                "visits": V[rows, cols],
            }
        )
        visits_df.to_csv(out / "visits.csv", index=False)
        files["visits"] = str(out / "visits.csv")

        # Config
        config_dict = {k: str(v) for k, v in self.config.__dict__.items()}
        pd.DataFrame([config_dict]).to_csv(out / "config.csv", index=False)
        files["config"] = str(out / "config.csv")

        print(f"\nResults saved to {out}/")
        return files


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_run(
    city_type: str = "polycentric",
    n_residents: int = 2000,
    n_pois: int = 200,
    n_timesteps: int = 50,
    **kwargs,
) -> MobilitySimulation:
    """
    Quick simulation run with sensible defaults.

    Good for testing and exploration.

    Args:
        city_type: Type of synthetic city to generate
        n_residents: Number of agents
        n_pois: Number of POIs
        n_timesteps: Simulation duration
        **kwargs: Additional SimulationConfig parameters

    Returns:
        Completed MobilitySimulation instance
    """
    config = SimulationConfig(
        n_residents=n_residents,
        n_pois=n_pois,
        n_timesteps=n_timesteps,
        **kwargs,
    )

    sim = MobilitySimulation(config)
    sim.setup(city_type)
    sim.run()

    return sim


def compare_models(
    city_type: str = "polycentric",
    n_residents: int = 2000,
    n_pois: int = 200,
    n_timesteps: int = 50,
    seed: int = 42,
) -> dict[str, MobilitySimulation]:
    """
    Run simulation with different destination models for comparison.

    Args:
        city_type: Type of synthetic city
        n_residents: Number of agents
        n_pois: Number of POIs
        n_timesteps: Simulation duration
        seed: Random seed

    Returns:
        Dict mapping model names to completed simulations
    """
    results = {}

    models = [
        DestinationModel.RANK_DISTANCE,
        DestinationModel.GRAVITY,
        DestinationModel.INTERVENING_OPPORTUNITIES,
    ]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Running with {model.value} model")
        print("=" * 60)

        config = SimulationConfig(
            destination_model=model,
            n_residents=n_residents,
            n_pois=n_pois,
            n_timesteps=n_timesteps,
            seed=seed,
        )

        sim = MobilitySimulation(config)
        sim.setup(city_type)
        sim.run(verbose=False)

        results[model.value] = sim

    return results


def compare_mechanisms(
    city_type: str = "polycentric",
    n_residents: int = 2000,
    n_pois: int = 200,
    n_timesteps: int = 50,
    seed: int = 42,
) -> dict[str, MobilitySimulation]:
    """
    Compare different mechanism toggles (EPR, recency, capacity).

    Args:
        city_type: Type of synthetic city
        n_residents: Number of agents
        n_pois: Number of POIs
        n_timesteps: Simulation duration
        seed: Random seed

    Returns:
        Dict mapping mechanism names to completed simulations
    """
    configs = {
        "baseline": SimulationConfig(
            use_epr=False,
            use_recency=False,
            enforce_capacity=False,
            n_residents=n_residents,
            n_pois=n_pois,
            n_timesteps=n_timesteps,
            seed=seed,
        ),
        "epr_only": SimulationConfig(
            use_epr=True,
            use_recency=False,
            enforce_capacity=False,
            n_residents=n_residents,
            n_pois=n_pois,
            n_timesteps=n_timesteps,
            seed=seed,
        ),
        "epr_recency": SimulationConfig(
            use_epr=True,
            use_recency=True,
            enforce_capacity=False,
            n_residents=n_residents,
            n_pois=n_pois,
            n_timesteps=n_timesteps,
            seed=seed,
        ),
        "full_model": SimulationConfig(
            use_epr=True,
            use_recency=True,
            enforce_capacity=True,
            n_residents=n_residents,
            n_pois=n_pois,
            n_timesteps=n_timesteps,
            seed=seed,
        ),
    }

    results = {}
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  {config.describe()}")
        print("=" * 60)

        sim = MobilitySimulation(config)
        sim.setup(city_type)
        sim.run(verbose=False)
        results[name] = sim

    return results
