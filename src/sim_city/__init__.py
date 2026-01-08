"""sim-city: Urban Mobility Simulation Package.

A framework for simulating urban mobility patterns with empirically-validated
models and computing segregation/isolation metrics.

Quick Start:
    >>> from sim_city import quick_run
    >>> sim = quick_run(n_agents=100, n_steps=50)
    >>> print(f"Mean isolation: {sim.isolation_stats['mean_isolation']:.3f}")

For more control:
    >>> from sim_city import SimulationConfig, MobilitySimulation
    >>> config = SimulationConfig(n_agents=500, n_steps=100, city_type="polycentric")
    >>> sim = MobilitySimulation(config)
    >>> sim.run()
"""

from sim_city.core import (
    DestinationModel,
    SimulationConfig,
    SimulationSnapshot,
    EvolutionTracker,
    MobilitySimulation,
    quick_run,
    compare_models,
    compare_mechanisms,
)

from sim_city.agents import (
    Demographics,
    AgentState,
    Agent,
    Population,
)

from sim_city.spatial import (
    Location,
    Zone,
    SpatialEnvironment,
    SyntheticCity,
    RealCity,
)

from sim_city.models import (
    DestinationChoiceModel,
    RankDistanceModel,
    GravityModel,
    InterveningOpportunitiesModel,
    EPREngine,
    epr_decision,
    apply_income_effect,
    destination_choice,
    select_return_location,
    enforce_capacity,
)

from sim_city.metrics import (
    SimulationMetrics,
    MetricsCalculator,
    compute_metrics,
    radius_of_gyration,
    experienced_isolation,
    exposure_matrix,
    fit_zipf,
    fit_truncated_powerlaw,
)

from sim_city.viz import (
    DiagnosticsSuite,
    plot_diagnostics,
    plot_evolution,
    plot_spatial,
    plot_distributions,
    plot_evolution_comparison,
    COLORS,
    ZONE_COLORS,
)

from sim_city.parameters import (
    LiteratureParameter,
    ParameterRegistry,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core
    "DestinationModel",
    "SimulationConfig",
    "SimulationSnapshot",
    "EvolutionTracker",
    "MobilitySimulation",
    "quick_run",
    "compare_models",
    "compare_mechanisms",
    # Agents
    "Demographics",
    "AgentState",
    "Agent",
    "Population",
    # Spatial
    "Location",
    "Zone",
    "SpatialEnvironment",
    "SyntheticCity",
    "RealCity",
    # Models
    "DestinationChoiceModel",
    "RankDistanceModel",
    "GravityModel",
    "InterveningOpportunitiesModel",
    "EPREngine",
    "epr_decision",
    "apply_income_effect",
    "destination_choice",
    "select_return_location",
    "enforce_capacity",
    # Metrics
    "SimulationMetrics",
    "MetricsCalculator",
    "compute_metrics",
    "radius_of_gyration",
    "experienced_isolation",
    "exposure_matrix",
    "fit_zipf",
    "fit_truncated_powerlaw",
    # Visualization
    "DiagnosticsSuite",
    "plot_diagnostics",
    "plot_evolution",
    "plot_spatial",
    "plot_distributions",
    "plot_evolution_comparison",
    "COLORS",
    "ZONE_COLORS",
    # Parameters
    "LiteratureParameter",
    "ParameterRegistry",
]
