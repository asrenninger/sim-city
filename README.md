# sim-city

Simulations and null models for urban science.

A Python package for simulating human mobility patterns in cities. The package supports two modes:

1. **Synthetic mode**: Generate counterfactual cities with configurable spatial structures and simulate how residents distribute trips among points of interest
2. **Real data mode**: Ingest actual POI data (e.g., from Overture, OpenStreetMap) and population data (Census, GHS-POP) to model mobility in real cities

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sim-city.git
cd sim-city

# Create environment
conda env create -f environment.yaml
conda activate sim-city

# Install in development mode
pip install -e ".[all]"
```

## Quickstart

```python
from sim_city import SimulationConfig, MobilitySimulation
from sim_city.spatial import SyntheticCity

# Configure simulation
config = SimulationConfig(
    n_residents=5_000,
    n_timesteps=100,
    destination_model='rank_distance',
    use_epr=True,
    enforce_capacity=True,
    seed=42
)

# Generate synthetic city
city = SyntheticCity(
    city_type='polycentric',
    extent=10.0,
    n_zones=5,
    n_pois=500
)

# Run simulation
sim = MobilitySimulation(config)
sim.load_city(city)
sim.run()

# Analyze
print(f"Mean isolation: {sim.metrics.mean_isolation:.3f}")
print(f"Mean Rg: {sim.metrics.radius_of_gyration.mean():.2f} km")
sim.plot_diagnostics()
```

## Development

```bash
make install   # Install in dev mode
make test      # Run tests
make lint      # Run linters
make format    # Format code
make notebook  # Launch Jupyter
```

## License

MIT License - see [LICENSE](LICENSE) for details.
