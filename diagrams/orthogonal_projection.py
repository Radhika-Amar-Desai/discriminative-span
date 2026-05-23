import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# RANDOM SEED
# =========================================================

np.random.seed(42)

# =========================================================
# CREATE FIGURE
# =========================================================

fig = plt.figure(figsize=(13, 10))

ax = fig.add_subplot(111, projection='3d')

# =========================================================
# DEFINE span(D) = x-y plane
# =========================================================

xx, yy = np.meshgrid(
    np.linspace(-2.5, 2.5, 10),
    np.linspace(-2.5, 2.5, 10)
)

zz = np.zeros_like(xx)

ax.plot_surface(
    xx,
    yy,
    zz,

    alpha=0.15,
    color='dodgerblue'
)

# =========================================================
# DRAW DIFFERENCE VECTORS
#
# Small vectors distributed across plane
# =========================================================

n_vectors = 10

for _ in range(n_vectors):

    start_x = np.random.uniform(-1.8, 1.8)
    start_y = np.random.uniform(-1.8, 1.8)

    dx = np.random.uniform(0.5, 1.0)
    dy = np.random.uniform(0.3, 0.8)

    dz = 0

    ax.quiver(
        start_x,
        start_y,
        0,

        dx,
        dy,
        dz,

        color='black',
        linewidth=1.4,
        alpha=0.55,
        arrow_length_ratio=0.08
    )

# =========================================================
# DEFINE TRUE CLASSIFIER NORMAL w
# =========================================================

w = np.array([
    1.5,
    1.0,
    2.2
])

w = w / np.linalg.norm(w)

w_scale = 3.2

w_end = w * w_scale

# =========================================================
# DRAW w
# =========================================================

ax.quiver(
    0,
    0,
    0,

    w_end[0],
    w_end[1],
    w_end[2],

    color='purple',
    linewidth=4.5,
    arrow_length_ratio=0.08
)

# =========================================================
# PROJECTION ONTO x-y plane
# =========================================================

projection = np.array([
    w_end[0],
    w_end[1],
    0
])

# =========================================================
# DRAW PROJECTION VECTOR
# =========================================================

ax.quiver(
    0,
    0,
    0,

    projection[0],
    projection[1],
    projection[2],

    color='crimson',
    linewidth=4,
    arrow_length_ratio=0.08
)

# =========================================================
# DRAW ORTHOGONAL ERROR
# =========================================================

ax.plot(
    [projection[0], w_end[0]],
    [projection[1], w_end[1]],
    [projection[2], w_end[2]],

    linestyle='--',
    linewidth=3,
    color='seagreen'
)

# =========================================================
# COMPUTE RELATIVE PROJECTION ERROR
# =========================================================

relative_error = (
    np.linalg.norm(w_end - projection)
    / np.linalg.norm(w_end)
)

# =========================================================
# LABELS
# =========================================================

# w label
ax.text(
    w_end[0] + 0.12,
    w_end[1] + 0.12,
    w_end[2] + 0.15,

    r'$\mathbf{w}$',

    fontsize=24,
    color='purple',
    weight='bold'
)

# Projection label
ax.text(
    projection[0] + 0.25,
    projection[1] - 0.4,
    0.12,

    r'$\Pi_{\mathrm{span}(D)}(\mathbf{w})$',

    fontsize=16,
    color='crimson'
)

# Error label
mid = (projection + w_end) / 2

ax.text(
    mid[0] + 0.12,
    mid[1] + 0.12,
    mid[2] + 0.2,

    r'$\mathbf{w}-\Pi_{\mathrm{span}(D)}(\mathbf{w})$',

    fontsize=15,
    color='seagreen'
)

# span(D) label
ax.text(
    -2.2,
    2.0,
    0.02,

    r'$\mathrm{span}(D)$',

    fontsize=22,
    color='dodgerblue',
    weight='bold'
)

# =========================================================
# FORMULA BOX
# =========================================================

formula_text = (
    r'$\mathrm{Relative\ Projection\ Error}$'
    '\n\n'
    r'$='
    r'\frac{\|\mathbf{w}-\Pi_{\mathrm{span}(D)}(\mathbf{w})\|}'
    r'{\|\mathbf{w}\|}$'
    '\n\n'
    f'$= {relative_error:.3f}$'
)

fig.text(
    0.80,
    0.33,

    formula_text,

    fontsize=15,

    bbox=dict(
        facecolor='white',
        alpha=0.96,
        edgecolor='gray',
        pad=14
    )
)

# =========================================================
# AXES LABELS
# =========================================================

ax.set_xlabel(
    'Embedding Dimension 1',
    fontsize=13,
    labelpad=12
)

ax.set_ylabel(
    'Embedding Dimension 2',
    fontsize=13,
    labelpad=12
)

ax.set_zlabel(
    'Embedding Dimension 3',
    fontsize=13,
    labelpad=12
)

# =========================================================
# TITLE
# =========================================================

ax.set_title(
    'Orthogonal Projection of Classifier Normal\n'
    'onto Difference Vector Span',

    fontsize=20,
    pad=28
)

# =========================================================
# VIEW
# =========================================================

ax.view_init(
    elev=24,
    azim=-58
)

# =========================================================
# LIMITS
# =========================================================

ax.set_xlim(-2.5, 3.5)
ax.set_ylim(-2.5, 3.5)
ax.set_zlim(0, 4)

# =========================================================
# GRID
# =========================================================

ax.grid(True)

# =========================================================
# LAYOUT
# =========================================================

fig.subplots_adjust(
    top=0.90,
    right=0.88
)

plt.tight_layout(
    rect=[0, 0, 0.88, 0.94]
)

plt.show()