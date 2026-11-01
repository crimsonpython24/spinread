import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spinread import plotstyle as st

st.apply()
figure = plt.figure(figsize=(st.COLUMN_WIDTH, 3.1))
grid = figure.add_gridspec(2, 2, hspace=0.05, wspace=0.05)
panels = [
    figure.add_subplot(grid[0, 0]),
    figure.add_subplot(grid[0, 1]),
    figure.add_subplot(grid[1, :]),
]
titles = [
    (r"topspin / backspin  $\omega_{\tilde y}$", "ball dips or floats"),
    (r"sidespin  $\omega_{\tilde z}$", "ball curves sideways"),
    (
        r"gyrospin  $\omega_{\tilde x}$",
        "no bend in flight; sideways kick at the bounce",
    ),
]
for index, (panel, (title, note)) in enumerate(zip(panels, titles)):
    panel.set_aspect("equal")
    panel.set_xlim(
        (-1.9, 1.9) if index < 2 else (-4.0, 4.0)
    )  # same ball size in the wide panel
    panel.set_ylim(-1.55, 1.5)
    panel.axis("off")
    panel.add_patch(Circle((0, 0), 0.75, fc="0.93", ec="black", lw=0.8))
    panel.add_patch(
        FancyArrowPatch(
            (0.8, 0), (1.75, 0), arrowstyle="-|>", mutation_scale=10, lw=1.0
        )
    )
    panel.text(
        1.28,
        0.12,
        r"$\mathbf{v}$, $\tilde x$",
        ha="center",
        va="bottom",
        fontsize=st.FONT,
    )
    panel.set_title(title, fontsize=st.TITLE, pad=2)
    panel.text(0, -1.32, note, ha="center", va="center", fontsize=st.FONT)
# topspin: axis y~ into the page; the ball rolls forward
panel = panels[0]
panel.add_patch(Arc((0, 0), 1.2, 1.2, theta1=200, theta2=520, lw=1.2, color=st.BLUE))
panel.add_patch(
    FancyArrowPatch(
        (0.59, 0.11),
        (0.61, -0.1),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.2,
        color=st.BLUE,
    )
)
panel.text(0, 0, r"$\otimes\,\tilde y$", ha="center", va="center", fontsize=st.TITLE)
# sidespin: axis z~ vertical; the ball turns about the vertical
panel = panels[1]
panel.add_patch(
    FancyArrowPatch(
        (0, 0), (0, 1.35), arrowstyle="-|>", mutation_scale=10, lw=1.0, linestyle="--"
    )
)
panel.text(0.12, 1.12, r"$\tilde z$", fontsize=st.FONT)
panel.add_patch(Arc((0, 0), 1.5, 0.5, theta1=190, theta2=350, lw=1.2, color=st.BLUE))
panel.add_patch(
    Arc(
        (0, 0),
        1.5,
        0.5,
        theta1=10,
        theta2=170,
        lw=1.2,
        color=st.BLUE,
        linestyle=(0, (2, 2)),
    )
)
panel.add_patch(
    FancyArrowPatch(
        (0.7, -0.12),
        (0.76, 0.02),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.2,
        color=st.BLUE,
    )
)
# gyrospin: axis x~ along the flight; the ball spins like a bullet
panel = panels[2]
panel.add_patch(Arc((0, 0), 0.5, 1.5, theta1=100, theta2=260, lw=1.2, color=st.BLUE))
panel.add_patch(
    Arc(
        (0, 0),
        0.5,
        1.5,
        theta1=280,
        theta2=440,
        lw=1.2,
        color=st.BLUE,
        linestyle=(0, (2, 2)),
    )
)
panel.add_patch(
    FancyArrowPatch(
        (-0.12, -0.7),
        (0.05, -0.74),
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.2,
        color=st.BLUE,
    )
)
st.finish(
    figure,
    "results/fig_spin_axes.png",
    left=0.02,
    right=0.98,
    top=0.93,
    bottom=0.03,
    hspace=0.5,
    wspace=0.05,
)
print("Wrote results/fig_spin_axes.png")
