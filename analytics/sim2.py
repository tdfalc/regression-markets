from pathlib import Path
import os
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

COLORS = cycle(["magenta", "blue", "darkorange", "limegreen"])

COLORS = cycle(
    [
        "#000000",  # Black
        "#F5793A",  # Orange
        "#4BA6FB",  # Modified blue (original was #85C0F9)
        "#A95AA1",  # Pink
        "#689948",  # Green: not optimal for colour-blind people
    ]
)


def calculate_price(intercept: float, a1: float, a2: float) -> float:
    return intercept - a1 - a2


def plot_externality(externalities: np.ndarray, savedir: Path) -> None:
    fig, ax = plt.subplots()

    cmap = "viridis"
    im = ax.imshow(externalities[::-1, :], cmap=cmap, extent=[0, 1, 0, 1])
    cbar = fig.colorbar(im, cmap=cmap, aspect=10)
    cbar.set_label("$\pi^{M} - \pi^{D}$", labelpad=20)  # rotation=270
    ax.set_xlabel("$a_i$")
    ax.set_ylabel("$a_j$")
    fig.tight_layout()
    fig.savefig(savedir / "cournot.pdf", dpi=300)


if __name__ == "__main__":

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12
    plt.rcParams["mathtext.fontset"] = "cm"  # Use CM for math font.
    plt.rcParams["figure.autolayout"] = True  # Use tight layouts.

    savedir = Path(__file__).parent / "docs/sim01_cournot_example"
    os.makedirs(savedir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 3), width_ratios=[1, 2], height_ratios=[1]
    )

    intercept = 5
    granularity = 1 / 1000
    xs = [0.1, 2]
    x_mean = np.mean(xs)
    x_cournot = intercept / 3
    assert x_mean < x_cournot

    # First we look at the externality from the perspective of player i when both
    # are subject to fundamental uncertainty.
    action_set = np.arange(0, x_mean + granularity, step=granularity)
    num_actions = len(action_set)
    externalities = np.zeros((num_actions, num_actions))

    for i, a1 in enumerate(action_set):
        price_monopoly = calculate_price(intercept, a1, 0)
        profit_monopoly = price_monopoly * a1

        for j, a2 in enumerate(action_set):
            price_duopoly = calculate_price(intercept, a1, a2)
            profit_duopoly = price_duopoly * a1
            externalities[j, i] = profit_monopoly - profit_duopoly

    cmap = "jet"
    im = ax1.imshow(externalities[::-1, :], cmap=cmap, extent=[0, 1, 0, 1])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cmap=cmap, cax=cax)
    cbar.set_label("Externality", labelpad=10)  # rotation=270
    ax1.set_xlabel("$a_i$")
    ax1.set_ylabel("$a_j$")

    # Next we explore the impact on social welfare and supplier surplus when player
    # i observes the state and possibly shares this information with the other.
    setups = {
        # "x=1/2": (x_mean, x_mean),
        # "x=1/2 (sharing)": (xs[0], xs[0]),
        "x=3/2": (xs[-1], x_mean),
        "x=3/2 (sharing)": (xs[-1], x_cournot),
    }
    for i, (name, (a1, a2)) in enumerate(setups.items()):
        action_set = np.arange(0, a1 + granularity, step=granularity)
        price = calculate_price(intercept, action_set, a2)
        profit = price * action_set
        idx = np.argmax(profit)
        total_quantity = action_set[idx] + a2
        supplier_surplus = profit + price * a2
        consumer_surplus = 0.5 * (a1 + a2) * (intercept - price)
        social_welfare = supplier_surplus + consumer_surplus
        color = next(COLORS)

        ax2.plot(
            action_set + a2,
            social_welfare,
            label=name,
            ls="dashed",
            lw=1,
            color=color,
            zorder=3 if i == 0 else 2,
        )
        ax2.plot(
            action_set + a2,
            supplier_surplus,
            label=None,
            ls="solid",
            lw=1,
            color=color,
            zorder=3 if i == 0 else 2,
        )

        ax2.scatter(
            [total_quantity, total_quantity],
            [social_welfare[idx], social_welfare[idx]],
            s=25,
            marker="o",
            lw=0.5,
            edgecolor="k",
            color=color,
            zorder=5,
        )

        ax2.scatter(
            [total_quantity, total_quantity],
            [supplier_surplus[idx], supplier_surplus[idx]],
            s=300,
            marker="*",
            lw=0.5,
            edgecolor="k",
            color=color,
            zorder=5,
        )

        ax2.plot(
            [total_quantity, total_quantity],
            [social_welfare[idx], supplier_surplus[idx]],
            lw=1,
            color=color,
            ls="dotted",
        )

    ax2.set_xlabel("Total Quantity ($a_i$ + $a_j$)")
    ax2.set_ylabel("Rent")
    ax2.legend(ncol=3, framealpha=1)

    leg = ax2.legend(
        facecolor="#eeeeee",
        edgecolor="#ffffff",
        framealpha=0.85,
        labelspacing=0.25,
        loc="upper left",
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_lw(1)
    ax2.spines["left"].set_lw(1)

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_tick_params(width=1)
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_tick_params(width=1)

    leg.get_frame().set_linewidth(0)
    fig.tight_layout()

    fig.savefig(savedir / "surplus.pdf", dpi=300)
