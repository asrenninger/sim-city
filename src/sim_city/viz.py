"""Visualization and diagnostics.

Publication-quality visualizations styled after The Economist / Financial Times.
Includes 12-panel diagnostic dashboards and standalone plot functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from sim_city.core import MobilitySimulation
    from sim_city.metrics import SimulationMetrics
    from sim_city.spatial import RealCity, SyntheticCity

# =============================================================================
# STYLE CONFIGURATION — Economist/FT aesthetic
# =============================================================================

COLORS = {
    "primary": "#1A5F7A",  # Deep teal
    "secondary": "#E63946",  # Muted red
    "tertiary": "#457B9D",  # Steel blue
    "quaternary": "#F4A261",  # Warm orange
    "quinary": "#2A9D8F",  # Sage green
    "dark": "#1D3557",  # Near black
    "mid": "#6C757D",  # Gray
    "light": "#ADB5BD",  # Light gray
    "bg": "#FFFFFF",  # White
    "grid": "#E9ECEF",  # Very light gray
}

ZONE_COLORS = {
    "Center": "#E63946",
    "NW": "#1A5F7A",
    "NE": "#2A9D8F",
    "SE": "#F4A261",
    "SW": "#457B9D",
}


def setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": COLORS["mid"],
            "axes.labelcolor": COLORS["dark"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "xtick.color": COLORS["mid"],
            "ytick.color": COLORS["mid"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


setup_style()


def _add_source_note(fig, text="Source: Mobility Simulation Framework", y=-0.02):
    """Add a source note at bottom of figure."""
    fig.text(
        0.02,
        y,
        text,
        fontsize=7,
        color=COLORS["light"],
        ha="left",
        va="top",
        style="italic",
    )


def _add_title_bar(ax, title, subtitle=None, y=1.08):
    """Add title with optional subtitle."""
    ax.text(
        0,
        y,
        title,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=COLORS["dark"],
        ha="left",
        va="bottom",
    )
    if subtitle:
        ax.text(
            0,
            y - 0.06,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color=COLORS["mid"],
            ha="left",
            va="bottom",
        )


def _format_axis(ax, xlabel=None, ylabel=None, grid="y"):
    """Clean axis formatting."""
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS["dark"])
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS["dark"])

    if grid == "y":
        ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.5, zorder=0)
    elif grid == "x":
        ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.5, zorder=0)
    elif grid == "both":
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, zorder=0)

    ax.set_axisbelow(True)


# =============================================================================
# DIAGNOSTICS SUITE
# =============================================================================


class DiagnosticsSuite:
    """
    Publication-quality diagnostics for mobility simulations.

    Example:
        sim = quick_run(...)
        diag = DiagnosticsSuite(sim)
        diag.plot_all()
    """

    def __init__(self, sim: "MobilitySimulation"):
        self.sim = sim
        self.config = sim.config

        # Pre-compute common data
        self.V = sim.metrics.compute_visit_matrix()
        self.rg_values = sim.metrics.compute_rg_distribution()

    def plot_all(self, save_path: str | None = None) -> plt.Figure:
        """Generate all diagnostic plots in a single figure."""
        # Temporarily disable interactive mode to prevent double display in notebooks
        was_interactive = plt.isinteractive()
        plt.ioff()

        try:
            fig = plt.figure(figsize=(16, 20))
            gs = GridSpec(
                4,
                3,
                figure=fig,
                hspace=0.35,
                wspace=0.25,
                left=0.06,
                right=0.94,
                top=0.94,
                bottom=0.04,
            )

            # Row 1: Network and Spatial
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_network_simple(ax1)

            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_poi_distribution(ax2)

            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_isolation_map(ax3)

            # Row 2: Evolution
            ax4 = fig.add_subplot(gs[1, 0])
            self._plot_isolation_evolution(ax4)

            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_location_accumulation(ax5)

            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_rg_evolution(ax6)

            # Row 3: Distributions
            ax7 = fig.add_subplot(gs[2, 0])
            self._plot_isolation_distribution(ax7)

            ax8 = fig.add_subplot(gs[2, 1])
            self._plot_zipf(ax8)

            ax9 = fig.add_subplot(gs[2, 2])
            self._plot_rg_distribution(ax9)

            # Row 4: Exposure and Zone comparison
            ax10 = fig.add_subplot(gs[3, 0])
            self._plot_exposure_heatmap(ax10)

            ax11 = fig.add_subplot(gs[3, 1])
            self._plot_zone_isolation_evolution(ax11)

            ax12 = fig.add_subplot(gs[3, 2])
            self._plot_summary_stats(ax12)

            # Main title
            config_str = self.config.describe()
            fig.suptitle(
                "Mobility Simulation Diagnostics",
                fontsize=16,
                fontweight="bold",
                color=COLORS["dark"],
                y=0.98,
            )
            fig.text(
                0.5, 0.955, config_str, ha="center", fontsize=9, color=COLORS["mid"]
            )

            _add_source_note(fig, y=0.01)

            if save_path:
                plt.savefig(save_path, dpi=300, facecolor="white")
                print(f"Saved to {save_path}")

            return fig

        finally:
            # Restore interactive mode if it was on
            if was_interactive:
                plt.ion()

    def _plot_network_simple(self, ax):
        """Simple network visualization."""
        rows, cols = np.nonzero(self.V)
        weights = self.V[rows, cols]

        top_n = min(2000, len(weights))
        top_idx = np.argsort(weights)[-top_n:]

        agent_coords = self.sim.population.agent_coords
        poi_coords = self.sim.spatial.poi_coords

        for idx in top_idx:
            i, j = rows[idx], cols[idx]
            w = weights[idx]
            alpha = min(0.3, 0.05 + 0.25 * (w / weights.max()))
            ax.plot(
                [agent_coords[i, 0], poi_coords[j, 0]],
                [agent_coords[i, 1], poi_coords[j, 1]],
                color=COLORS["primary"],
                alpha=alpha,
                linewidth=0.3,
                zorder=1,
            )

        ax.scatter(
            poi_coords[:, 0],
            poi_coords[:, 1],
            c=COLORS["secondary"],
            s=15,
            alpha=0.8,
            zorder=3,
            edgecolors="white",
            linewidth=0.5,
        )

        ax.set_aspect("equal")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.axis("off")
        _add_title_bar(ax, "Mobility Network", "Top flows by volume")

    def _plot_poi_distribution(self, ax):
        """POI locations colored by total visits."""
        poi_coords = self.sim.spatial.poi_coords
        visit_totals = self.V.sum(axis=0)

        scatter = ax.scatter(
            poi_coords[:, 0],
            poi_coords[:, 1],
            c=visit_totals,
            cmap="YlOrRd",
            s=30,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Total visits", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_aspect("equal")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.axis("off")
        _add_title_bar(ax, "POI Popularity", "Color = visit count")

    def _plot_isolation_map(self, ax):
        """Agent locations colored by experienced isolation."""
        agent_coords = self.sim.population.agent_coords
        isolation = self.sim.isolation_df["experienced_isolation"].values

        order = np.argsort(isolation)

        scatter = ax.scatter(
            agent_coords[order, 0],
            agent_coords[order, 1],
            c=isolation[order],
            cmap="RdYlBu_r",
            s=4,
            alpha=0.6,
            vmin=0,
            vmax=1,
        )

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Isolation", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_aspect("equal")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.axis("off")

        mean_iso = isolation.mean()
        _add_title_bar(ax, "Experienced Isolation", f"Mean = {mean_iso:.3f}")

    def _plot_isolation_evolution(self, ax):
        """Mean isolation over time with confidence band."""
        evo = self.sim.evolution_df

        ax.fill_between(
            evo["timestep"],
            evo["mean_isolation"] - evo["std_isolation"],
            evo["mean_isolation"] + evo["std_isolation"],
            alpha=0.2,
            color=COLORS["primary"],
            linewidth=0,
        )
        ax.plot(
            evo["timestep"], evo["mean_isolation"], color=COLORS["primary"], linewidth=2
        )

        final = evo["mean_isolation"].iloc[-1]
        ax.annotate(
            f"{final:.3f}",
            xy=(evo["timestep"].iloc[-1], final),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
            color=COLORS["primary"],
            fontweight="bold",
            va="center",
        )

        _format_axis(ax, xlabel="Timestep", ylabel="Mean isolation")
        _add_title_bar(ax, "Isolation", "Convergence to equilibrium")

    def _plot_location_accumulation(self, ax):
        """Unique locations over time with confidence band."""
        evo = self.sim.evolution_df

        timesteps = evo["timestep"].values
        mean_locs = evo["mean_unique_locations"].values

        if "std_unique_locations" in evo.columns:
            std_locs = evo["std_unique_locations"].values
            ax.fill_between(
                timesteps,
                mean_locs - std_locs,
                mean_locs + std_locs,
                color=COLORS["tertiary"],
                alpha=0.2,
            )

        ax.plot(timesteps, mean_locs, color=COLORS["tertiary"], linewidth=2)

        capacity = self.config.location_capacity_mean
        ax.axhline(
            capacity, color=COLORS["secondary"], linestyle="--", linewidth=1.5, alpha=0.7
        )
        ax.text(
            timesteps[0],
            capacity + 0.5,
            f"Capacity (L={capacity:.0f})",
            fontsize=8,
            color=COLORS["secondary"],
            va="bottom",
        )

        _format_axis(ax, xlabel="Timestep", ylabel="Mean unique locations")
        _add_title_bar(ax, "Location Accumulation", "Alessandretti capacity constraint")

    def _plot_rg_evolution(self, ax):
        """Radius of gyration over time with confidence band."""
        evo = self.sim.evolution_df

        timesteps = evo["timestep"].values
        mean_rg = evo["mean_rg"].values

        if "std_rg" in evo.columns:
            std_rg = evo["std_rg"].values
            ax.fill_between(
                timesteps,
                mean_rg - std_rg,
                mean_rg + std_rg,
                color=COLORS["quinary"],
                alpha=0.2,
            )

        ax.plot(timesteps, mean_rg, color=COLORS["quinary"], linewidth=2)

        _format_axis(ax, xlabel="Timestep", ylabel="Mean Rg")
        _add_title_bar(ax, "Spatial Extent", "Radius of gyration")

    def _plot_isolation_distribution(self, ax):
        """Histogram of experienced isolation."""
        isolation = self.sim.isolation_df["experienced_isolation"]

        ax.hist(
            isolation,
            bins=30,
            density=True,
            alpha=0.7,
            color=COLORS["primary"],
            edgecolor="white",
            linewidth=0.5,
        )

        mean_val = isolation.mean()
        ax.axvline(mean_val, color=COLORS["secondary"], linewidth=2, linestyle="-")
        ax.text(
            mean_val + 0.02,
            ax.get_ylim()[1] * 0.9,
            f"μ = {mean_val:.3f}",
            fontsize=9,
            color=COLORS["secondary"],
            fontweight="bold",
            va="top",
        )

        ax.set_xlim(0, 1)
        _format_axis(ax, xlabel="Experienced isolation", ylabel="Density")
        _add_title_bar(ax, "Isolation Distribution", "Cross-sectional")

    def _plot_zipf(self, ax):
        """Rank-frequency plot (Zipf check)."""
        visit_counts = self.V.sum(axis=0)
        visit_counts = visit_counts[visit_counts > 0]

        sorted_counts = np.sort(visit_counts)[::-1]
        ranks = np.arange(1, len(sorted_counts) + 1)

        ax.loglog(
            ranks, sorted_counts, "o", color=COLORS["primary"], markersize=4, alpha=0.6
        )

        if len(ranks) > 1:
            x_fit = np.linspace(1, len(ranks), 100)
            y_fit = sorted_counts[0] * x_fit ** (-1)
            ax.loglog(
                x_fit,
                y_fit,
                "--",
                color=COLORS["secondary"],
                linewidth=1.5,
                label="Zipf (ζ = −1)",
            )
            ax.legend(frameon=False, loc="lower left")

        _format_axis(ax, xlabel="Rank", ylabel="Visit count", grid="both")
        _add_title_bar(ax, "Zipf Law Check", "POI visitation frequency")

    def _plot_rg_distribution(self, ax):
        """Histogram of radius of gyration."""
        rg = self.rg_values[self.rg_values > 0]

        ax.hist(
            rg,
            bins=30,
            density=True,
            alpha=0.7,
            color=COLORS["quinary"],
            edgecolor="white",
            linewidth=0.5,
        )

        mean_val = rg.mean()
        ax.axvline(mean_val, color=COLORS["secondary"], linewidth=2)
        ax.text(
            mean_val + 0.2,
            ax.get_ylim()[1] * 0.9,
            f"μ = {mean_val:.2f}",
            fontsize=9,
            color=COLORS["secondary"],
            fontweight="bold",
            va="top",
        )

        _format_axis(ax, xlabel="Radius of gyration", ylabel="Density")
        _add_title_bar(ax, "Rg Distribution", "Spatial footprint")

    def _plot_exposure_heatmap(self, ax):
        """Exposure matrix heatmap."""
        exposure = self.sim.exposure_df.set_index("origin_zone")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "exposure", ["#FFFFFF", COLORS["primary"]]
        )

        im = ax.imshow(exposure.values, cmap=cmap, vmin=0, vmax=0.6, aspect="equal")

        zones = list(exposure.index)
        ax.set_xticks(range(len(zones)))
        ax.set_yticks(range(len(zones)))
        ax.set_xticklabels(zones, fontsize=9)
        ax.set_yticklabels(zones, fontsize=9)
        ax.set_xlabel("Encountered zone", fontsize=9)
        ax.set_ylabel("Origin zone", fontsize=9)

        for i in range(len(zones)):
            for j in range(len(zones)):
                val = exposure.values[i, j]
                color = "white" if val > 0.35 else COLORS["dark"]
                weight = "bold" if i == j else "normal"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight=weight,
                )

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Exposure share", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        _add_title_bar(ax, "Exposure Matrix", "Diagonal = isolation")

    def _plot_zone_isolation_evolution(self, ax):
        """Zone-level isolation over time."""
        evo = self.sim.evolution_df

        zone_cols = [c for c in evo.columns if c.startswith("isolation_")]

        for col in zone_cols:
            zone = col.replace("isolation_", "")
            color = ZONE_COLORS.get(zone, COLORS["mid"])
            ax.plot(evo["timestep"], evo[col], color=color, linewidth=1.5, label=zone)

        ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
        _format_axis(ax, xlabel="Timestep", ylabel="Mean isolation")
        _add_title_bar(ax, "Zone Isolation", "By residential zone")

    def _plot_summary_stats(self, ax):
        """Summary statistics panel."""
        ax.axis("off")

        iso = self.sim.isolation_df["experienced_isolation"]
        evo = self.sim.evolution_df

        stats = [
            ("Configuration", ""),
            ("Residents", f"{self.config.n_residents:,}"),
            ("POIs", f"{self.config.n_pois:,}"),
            ("Timesteps", f"{self.config.n_timesteps}"),
            ("", ""),
            ("Isolation", ""),
            ("Mean", f"{iso.mean():.3f}"),
            ("Std", f"{iso.std():.3f}"),
            ("Min / Max", f"{iso.min():.3f} / {iso.max():.3f}"),
            ("", ""),
            ("Final State", ""),
            ("Locations/person", f'{evo["mean_unique_locations"].iloc[-1]:.1f}'),
            ("Mean Rg", f'{evo["mean_rg"].iloc[-1]:.2f}'),
            ("Total visits", f"{int(self.V.sum()):,}"),
            ("", ""),
            ("Mechanisms", ""),
            ("EPR", "Yes" if self.config.use_epr else "No"),
            ("Recency", "Yes" if self.config.use_recency else "No"),
            ("Capacity", "Yes" if self.config.enforce_capacity else "No"),
        ]

        y = 0.95
        for label, value in stats:
            if value == "":
                ax.text(
                    0.05,
                    y,
                    label,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    color=COLORS["dark"],
                    va="top",
                )
            else:
                ax.text(
                    0.08,
                    y,
                    label,
                    transform=ax.transAxes,
                    fontsize=9,
                    color=COLORS["mid"],
                    va="top",
                )
                ax.text(
                    0.55,
                    y,
                    value,
                    transform=ax.transAxes,
                    fontsize=9,
                    color=COLORS["dark"],
                    va="top",
                    fontweight="bold",
                )
            y -= 0.05

        _add_title_bar(ax, "Summary", "Key metrics")


# =============================================================================
# STANDALONE PLOT FUNCTIONS
# =============================================================================


def plot_diagnostics(
    sim: "MobilitySimulation",
    figsize: tuple[float, float] = (16, 12),
    style: str = "publication",
    save_path: str | None = None,
) -> plt.Figure:
    """Generate 12-panel diagnostic dashboard."""
    diag = DiagnosticsSuite(sim)
    return diag.plot_all(save_path=save_path)


def plot_evolution(
    evolution_df: pd.DataFrame,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Plot evolution time series."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Mean isolation
    ax = axes[0, 0]
    ax.plot(
        evolution_df["timestep"],
        evolution_df["mean_isolation"],
        color=COLORS["primary"],
        linewidth=2,
    )
    ax.fill_between(
        evolution_df["timestep"],
        evolution_df["mean_isolation"] - evolution_df["std_isolation"],
        evolution_df["mean_isolation"] + evolution_df["std_isolation"],
        alpha=0.2,
        color=COLORS["primary"],
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Isolation")
    ax.set_title("Isolation Evolution")
    ax.grid(True, alpha=0.3)

    # Unique locations
    ax = axes[0, 1]
    ax.plot(
        evolution_df["timestep"],
        evolution_df["mean_unique_locations"],
        color=COLORS["tertiary"],
        linewidth=2,
    )
    ax.axhline(25, color=COLORS["secondary"], linestyle="--", label="Capacity L=25")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Unique Locations")
    ax.set_title("Location Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Radius of gyration
    ax = axes[1, 0]
    ax.plot(
        evolution_df["timestep"],
        evolution_df["mean_rg"],
        color=COLORS["quinary"],
        linewidth=2,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Rg")
    ax.set_title("Spatial Extent")
    ax.grid(True, alpha=0.3)

    # Zone isolation
    ax = axes[1, 1]
    zone_cols = [c for c in evolution_df.columns if c.startswith("isolation_")]
    for col in zone_cols:
        zone = col.replace("isolation_", "")
        color = ZONE_COLORS.get(zone, COLORS["mid"])
        ax.plot(evolution_df["timestep"], evolution_df[col], color=color, label=zone)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Isolation")
    ax.set_title("Zone Isolation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spatial(sim: "MobilitySimulation", figsize: tuple[float, float] = (16, 5)):
    """Plot spatial patterns."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Agent homes
    ax = axes[0]
    for zone in ZONE_COLORS:
        mask = sim.agent_df["zone_name"] == zone
        ax.scatter(
            sim.agent_df.loc[mask, "home_x"],
            sim.agent_df.loc[mask, "home_y"],
            c=ZONE_COLORS[zone],
            s=3,
            alpha=0.5,
            label=zone,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Agent Homes")
    ax.legend(markerscale=3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # POIs
    ax = axes[1]
    scatter = ax.scatter(
        sim.poi_df["x"],
        sim.poi_df["y"],
        c=sim.poi_df["attractiveness"],
        cmap="YlOrRd",
        s=25,
        alpha=0.8,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("POI Locations")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Attractiveness")

    # Isolation map
    ax = axes[2]
    scatter = ax.scatter(
        sim.agent_df["home_x"],
        sim.agent_df["home_y"],
        c=sim.isolation_df["experienced_isolation"],
        cmap="RdYlBu_r",
        s=5,
        alpha=0.6,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Isolation Map")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Isolation")

    plt.tight_layout()
    return fig


def plot_distributions(sim: "MobilitySimulation", figsize: tuple[float, float] = (15, 4)):
    """Plot key distributions."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    V = sim.metrics.compute_visit_matrix()
    rg_values = sim.metrics.compute_rg_distribution()

    # Isolation distribution
    ax = axes[0]
    ax.hist(
        sim.isolation_df["experienced_isolation"],
        bins=30,
        density=True,
        alpha=0.7,
        color=COLORS["primary"],
    )
    mean_val = sim.isolation_df["experienced_isolation"].mean()
    ax.axvline(mean_val, color=COLORS["secondary"], linestyle="--", lw=2)
    ax.set_xlabel("Experienced Isolation")
    ax.set_ylabel("Density")
    ax.set_title(f"Isolation Distribution (μ={mean_val:.3f})")
    ax.grid(True, alpha=0.3)

    # Zipf check
    ax = axes[1]
    visit_counts = V.sum(axis=0)
    visit_counts = visit_counts[visit_counts > 0]
    sorted_counts = np.sort(visit_counts)[::-1]
    ranks = np.arange(1, len(sorted_counts) + 1)
    ax.loglog(ranks, sorted_counts, "ko", markersize=3, alpha=0.5)
    x_fit = np.linspace(1, len(ranks), 100)
    y_fit = sorted_counts[0] * x_fit ** (-1)
    ax.loglog(x_fit, y_fit, "r--", label="Zipf (ζ=-1)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Visits")
    ax.set_title("Zipf Check")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rg distribution
    ax = axes[2]
    rg = rg_values[rg_values > 0]
    ax.hist(rg, bins=30, density=True, alpha=0.7, color=COLORS["quinary"])
    mean_rg = rg.mean()
    ax.axvline(mean_rg, color=COLORS["secondary"], linestyle="--", lw=2)
    ax.set_xlabel("Radius of Gyration")
    ax.set_ylabel("Density")
    ax.set_title(f"Rg Distribution (μ={mean_rg:.2f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_city_map(
    city: "SyntheticCity | RealCity",
    color_by: str = "category",
    show_zones: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Spatial visualization of city structure."""
    raise NotImplementedError("Use plot_spatial() for now")


def plot_comparison(
    results: list["SimulationMetrics"],
    labels: list[str],
    metrics: list[str] = ["isolation", "rg", "entropy"],
) -> plt.Figure:
    """Compare multiple simulation runs."""
    raise NotImplementedError("Use plot_evolution_comparison() for now")


def plot_evolution_comparison(
    results: dict[str, "MobilitySimulation"], figsize: tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Compare evolution across multiple simulations.

    Args:
        results: Dict of {name: MobilitySimulation}
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = list(ZONE_COLORS.values())[: len(results)]

    for idx, (name, sim) in enumerate(results.items()):
        evo = sim.evolution_df
        color = colors[idx % len(colors)]

        axes[0].plot(
            evo["timestep"], evo["mean_isolation"], color=color, linewidth=2, label=name
        )
        axes[1].plot(
            evo["timestep"],
            evo["mean_unique_locations"],
            color=color,
            linewidth=2,
            label=name,
        )
        axes[2].plot(
            evo["timestep"], evo["mean_rg"], color=color, linewidth=2, label=name
        )

    axes[0].set_ylabel("Mean isolation")
    axes[1].set_ylabel("Mean locations")
    axes[2].set_ylabel("Mean Rg")

    for ax in axes:
        ax.set_xlabel("Timestep")
        ax.legend(frameon=False, fontsize=8)
        _format_axis(ax, grid="y")

    _add_title_bar(axes[0], "Isolation")
    _add_title_bar(axes[1], "Locations")
    _add_title_bar(axes[2], "Spatial Extent")

    fig.suptitle(
        "Evolution Comparison",
        fontsize=14,
        fontweight="bold",
        color=COLORS["dark"],
        y=1.05,
    )
    _add_source_note(fig)
    plt.tight_layout()
    return fig
