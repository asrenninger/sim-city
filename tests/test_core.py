"""Tests for core module."""

import pytest

from sim_city.core import MobilitySimulation, SimulationConfig


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SimulationConfig()
        assert config.n_residents == 10_000
        assert config.n_timesteps == 150
        assert config.destination_model == "rank_distance"
        assert config.rank_distance_alpha == 0.84
        assert config.use_epr is True
        assert config.epr_rho == 0.6
        assert config.epr_gamma == 0.21

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SimulationConfig(n_residents=5000, use_epr=False)
        assert config.n_residents == 5000
        assert config.use_epr is False


class TestMobilitySimulation:
    """Tests for MobilitySimulation."""

    def test_initialization(self):
        """Test simulation initialization."""
        config = SimulationConfig()
        sim = MobilitySimulation(config)
        assert sim.config == config
        assert sim.city is None
        assert sim.population is None
