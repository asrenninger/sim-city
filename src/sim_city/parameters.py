"""Literature-derived parameters with provenance tracking.

This module provides a registry of all parameters derived from empirical
mobility research, with their sources, uncertainties, and alternative values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LiteratureParameter:
    """A parameter with its provenance and uncertainty.

    Attributes:
        value: The default/recommended value
        source: Citation for the parameter source
        uncertainty: Optional uncertainty range (±)
        context: Description of when/how the parameter applies
        alternatives: Dict of alternative values with their sources
    """

    value: float
    source: str
    uncertainty: Optional[float] = None
    context: str = ""
    alternatives: Dict[str, float] = field(default_factory=dict)


class ParameterRegistry:
    """Central registry of all literature-derived parameters.

    This registry provides easy access to empirically-validated parameters
    from mobility research, with full provenance tracking.

    Example:
        >>> from sim_city.parameters import ParameterRegistry
        >>> alpha = ParameterRegistry.RANK_DISTANCE_ALPHA.value
        >>> print(f"Using α={alpha} from {ParameterRegistry.RANK_DISTANCE_ALPHA.source}")
    """

    # =========================================================================
    # DESTINATION CHOICE PARAMETERS
    # =========================================================================

    RANK_DISTANCE_ALPHA = LiteratureParameter(
        value=0.84,
        source="Noulas et al. (2012) 'A tale of many cities'",
        uncertainty=0.05,
        context="Rank exponent for distance decay in destination choice",
        alternatives={
            "Liang et al. (2015)": 0.88,
            "Yan et al. (2017)": 0.91,
        },
    )

    GRAVITY_BETA = LiteratureParameter(
        value=2.0,
        source="Zipf (1946), validated in modern mobility studies",
        uncertainty=0.3,
        context="Distance decay exponent in gravity models",
        alternatives={
            "Erlander & Stewart (1990)": 1.8,
            "Simini et al. (2012)": 2.2,
        },
    )

    INTERVENING_OPPORTUNITIES_ALPHA = LiteratureParameter(
        value=0.2,
        source="Stouffer (1940), extended by Ruiter (1967)",
        context="Selectivity parameter for intervening opportunities model",
        alternatives={
            "Schneider (1959)": 0.15,
        },
    )

    # =========================================================================
    # EPR (EXPLORATION-PREFERENTIAL RETURN) PARAMETERS
    # =========================================================================

    EPR_RHO = LiteratureParameter(
        value=0.6,
        source="Pappalardo et al. (2015), Song et al. (2010)",
        uncertainty=0.05,
        context="Probability of exploring new location vs returning",
        alternatives={
            "Schläpfer et al. (2021)": 0.55,
            "Alessandretti et al. (2018)": 0.65,
        },
    )

    EPR_GAMMA = LiteratureParameter(
        value=0.21,
        source="Pappalardo et al. (2015)",
        uncertainty=0.03,
        context="Controls how exploration probability decays with unique locations visited",
        alternatives={
            "Song et al. (2010)": 0.20,
        },
    )

    # =========================================================================
    # RECENCY PARAMETERS
    # =========================================================================

    RECENCY_WINDOW = LiteratureParameter(
        value=5,
        source="Barbosa et al. (2015)",
        context="Number of recent locations to consider for recency-weighted returns",
        alternatives={
            "Alessandretti et al. (2018)": 7,
        },
    )

    RECENCY_WEIGHT = LiteratureParameter(
        value=0.3,
        source="Barbosa et al. (2015)",
        uncertainty=0.1,
        context="Weight given to recency vs frequency in return decisions",
        alternatives={
            "Custom": 0.5,
        },
    )

    # =========================================================================
    # CAPACITY PARAMETERS
    # =========================================================================

    CAPACITY_L = LiteratureParameter(
        value=25,
        source="Alessandretti et al. (2018)",
        uncertainty=5,
        context="Typical number of locations in individual mobility repertoire",
        alternatives={
            "Pappalardo et al. (2015)": 20,
            "Song et al. (2010)": 30,
        },
    )

    # =========================================================================
    # INCOME EFFECT PARAMETERS
    # =========================================================================

    INCOME_DISTANCE_ELASTICITY = LiteratureParameter(
        value=0.15,
        source="Estimated from NHTS and ACS data",
        uncertainty=0.05,
        context="How much higher income increases travel distance",
    )

    INCOME_DIVERSITY_ELASTICITY = LiteratureParameter(
        value=0.10,
        source="Estimated from mobility data",
        uncertainty=0.05,
        context="How much higher income increases location diversity",
    )

    # =========================================================================
    # DISTRIBUTION PARAMETERS (for validation)
    # =========================================================================

    ZIPF_EXPONENT = LiteratureParameter(
        value=-1.0,
        source="Zipf's law for POI popularity",
        uncertainty=0.1,
        context="Expected Zipf exponent for location visit frequencies",
    )

    RG_BETA = LiteratureParameter(
        value=1.65,
        source="González et al. (2008)",
        uncertainty=0.15,
        context="Power law exponent for radius of gyration distribution",
    )

    RG_KAPPA = LiteratureParameter(
        value=100.0,
        source="González et al. (2008)",
        uncertainty=20.0,
        context="Exponential cutoff for radius of gyration (in km)",
    )

    @classmethod
    def get_all(cls) -> Dict[str, LiteratureParameter]:
        """Return all parameters as a dictionary."""
        return {
            name: value
            for name, value in vars(cls).items()
            if isinstance(value, LiteratureParameter)
        }

    @classmethod
    def summary(cls) -> str:
        """Return a formatted summary of all parameters."""
        lines = ["Literature Parameter Registry", "=" * 40]
        for name, param in cls.get_all().items():
            unc = f" (±{param.uncertainty})" if param.uncertainty else ""
            lines.append(f"\n{name}:")
            lines.append(f"  Value: {param.value}{unc}")
            lines.append(f"  Source: {param.source}")
            if param.context:
                lines.append(f"  Context: {param.context}")
            if param.alternatives:
                lines.append("  Alternatives:")
                for src, val in param.alternatives.items():
                    lines.append(f"    - {src}: {val}")
        return "\n".join(lines)
