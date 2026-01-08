"""Tests for metrics module."""

import pytest
import numpy as np

from sim_city.metrics import SimulationMetrics


class TestSimulationMetrics:
    """Tests for SimulationMetrics container."""

    def test_dataclass_creation(self):
        """Test metrics dataclass can be created."""
        metrics = SimulationMetrics(
            n_unique_locations=np.array([10, 15, 20]),
            radius_of_gyration=np.array([2.5, 3.0, 4.5]),
            entropy=np.array([1.5, 2.0, 2.5]),
            experienced_isolation=np.array([0.3, 0.4, 0.5]),
            mean_isolation=0.4,
            std_isolation=0.1,
            exposure_matrix=np.eye(5),
            zipf_exponent=1.2,
            rg_exponent=1.5,
        )
        assert metrics.mean_isolation == 0.4
        assert len(metrics.n_unique_locations) == 3


class TestRadiusOfGyration:
    """Tests for radius of gyration calculation."""

    def test_single_location(self):
        """Rg should be 0 for a single location."""
        # TODO: Implement when function is complete
        pass

    def test_symmetric_locations(self):
        """Test Rg for symmetric location distribution."""
        # TODO: Implement when function is complete
        pass


class TestExposureMatrix:
    """Tests for exposure matrix calculation."""

    def test_matrix_shape(self):
        """Exposure matrix should be n_zones x n_zones."""
        # TODO: Implement when function is complete
        pass

    def test_row_sums(self):
        """Exposure matrix rows should sum to 1."""
        # TODO: Implement when function is complete
        pass
