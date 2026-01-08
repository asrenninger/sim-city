"""Tests for spatial module."""

import pytest

from sim_city.spatial import RealCity, SyntheticCity


class TestSyntheticCity:
    """Tests for SyntheticCity generation."""

    def test_initialization(self):
        """Test city initialization."""
        city = SyntheticCity(
            city_type="polycentric", extent=10.0, n_zones=5, n_pois=500
        )
        assert city.city_type == "polycentric"
        assert city.extent == 10.0
        assert city.n_zones == 5
        assert city.n_pois == 500

    def test_city_types(self):
        """Test all supported city types can be instantiated."""
        city_types = ["monocentric", "polycentric", "urban_villages", "edge_city", "grid"]
        for city_type in city_types:
            city = SyntheticCity(city_type=city_type)
            assert city.city_type == city_type


class TestRealCity:
    """Tests for RealCity loading."""

    def test_initialization(self):
        """Test RealCity initialization."""
        city = RealCity()
        assert city._pois is None
        assert city._zones is None
