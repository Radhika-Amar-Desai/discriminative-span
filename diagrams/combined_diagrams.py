import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# =========================================================
# RANDOM SEED
# =========================================================

np.random.seed(42)

# =========================================================
# CREATE FIGURE
# =========================================================

fig = plt.figure(figsize=(14, 5.8))

# =========================================================
# =========================================================
# PANEL (a):
# EMBEDDING SPACE GEOMETRY
# =========================================================
# =========================================================

ax1 = fig.add_subplot(
    121,
    projection='3d'
)

# =========================================================
# GENERATE DATA
# =========================================================

n_points = 14

# ---------------------------------------------------------
# Real Class A
# ---------------------------------------------------------

x_a = np.random.normal(
    -0.5,
    0.45,
    n_points
)

y_a = np.random.normal(
    -0.8,
    0.45,
    n_points
)

z_a = np.random.normal(
    0.0,
    0.08,
    n_points
)

real_A = np.vstack([
    x_a,
    y_a,
    z_a
]).T

# ---------------------------------------------------------
# Synthetic Class B
# ---------------------------------------------------------

base_shift = np.array([
    2.6,
    2.0,
    0.12
])

direction_noise = np.random.normal(
    loc=0,
    scale=[
        0.18,
        0.22,
        0.015
    ],
    size=(n_points, 3)
)

synthetic_B = (
    real_A
    + base_shift
    + direction_noise
)

# ---------------------------------------------------------
# Real Class B
# ---------------------------------------------------------

x_b = np.random.normal(
    1.3,
    0.35,
    n_points
)

y_b = np.random.normal(
    1.2,
    0.35,
    n_points
)

z_b = np.random.normal(
    1.9,
    0.28,
    n_points
)

real_B = np.vstack([
    x_b,
    y_b,
    z_b
]).T

# =========================================================
# PLOT DATA
# =========================================================

ax1.scatter(
    real_A[:, 0],
    real_A[:, 1],
    real_A[:, 2],

    marker='*',
    s=180,
    color='royalblue',
    label='Real A'
)

ax1.scatter(
    synthetic_B[:, 0],
    synthetic_B[:, 1],
    synthetic_B[:, 2],

    marker='o',
    s=55,
    color='darkorange',
    edgecolors='black',
    label='Synthetic B'
)

ax1.scatter(
    real_B[:, 0],
    real_B[:, 1],
    real_B[:, 2],

    marker='o',
    s=55,
    color='forestgreen',
    edgecolors='black',
    label='Real B'
)

# =========================================================
# DIFFERENCE VECTORS
# =========================================================

D = synthetic_B - real_A

for i in range(n_points):

    start = real_A[i]
    direction = D[i]

    ax1.quiver(
        start[0],
        start[1],
        start[2],

        direction[0],
        direction[1],
        direction[2],

        arrow_length_ratio=0.08,
        color='black',
        linewidth=1.0,
        alpha=0.40
    )

# =========================================================
# SPAN PLANE
# =========================================================

pca = PCA(n_components=2)

pca.fit(D)

v1 = pca.components_[0]
v2 = pca.components_[1]

span_center = (
    np.mean(synthetic_B, axis=0)
    - np.array([1.4, 1.3, 0.0])
)

plane_range = np.linspace(
    -1.8,
    1.8,
    10
)

uu, vv = np.meshgrid(
    plane_range,
    plane_range
)

plane_points = (
    span_center
    + uu[..., np.newaxis] * v1
    + vv[..., np.newaxis] * v2
)

xx = plane_points[:, :, 0]
yy = plane_points[:, :, 1]
zz = plane_points[:, :, 2]

ax1.plot_surface(
    xx,
    yy,
    zz,

    alpha=0.15,
    color='dodgerblue'
)

# =========================================================
# TRAIN CLASSIFIER
# =========================================================

X = np.vstack([
    real_A,
    real_B
])

y = np.concatenate([
    np.zeros(len(real_A)),
    np.ones(len(real_B))
])

clf = LogisticRegression()

clf.fit(X, y)

w = clf.coef_[0]

w = w / np.linalg.norm(w)

b = clf.intercept_[0]

# =========================================================
# CLASSIFIER NORMAL VECTOR
# =========================================================

origin = np.mean(X, axis=0)

normal_scale = 2.3

ax1.quiver(
    origin[0],
    origin[1],
    origin[2],

    w[0] * normal_scale,
    w[1] * normal_scale,
    w[2] * normal_scale,

    color='purple',
    linewidth=3.8,
    arrow_length_ratio=0.08
)

ax1.text(
    origin[0] + w[0] * normal_scale,
    origin[1] + w[1] * normal_scale,
    origin[2] + w[2] * normal_scale + 0.10,

    r'$\mathbf{w}$',

    fontsize=18,
    color='purple',
    weight='bold'
)

# =========================================================
# CLASSIFIER HYPERPLANE
# =========================================================

plane_center = -b * w

plane_center = (
    plane_center
    - 0.6 * w
)

arb = np.array([1, 0, 0])

if np.abs(np.dot(arb, w)) > 0.9:
    arb = np.array([0, 1, 0])

u = np.cross(w, arb)
u = u / np.linalg.norm(u)

v = np.cross(w, u)
v = v / np.linalg.norm(v)

plane_size = 4.0

s = np.linspace(
    -plane_size,
    plane_size,
    10
)

t = np.linspace(
    -plane_size,
    plane_size,
    10
)

S, T = np.meshgrid(s, t)

plane_points = (
    plane_center
    + S[..., np.newaxis] * u
    + T[..., np.newaxis] * v
)

grid_x = plane_points[:, :, 0]
grid_y = plane_points[:, :, 1]
grid_z = plane_points[:, :, 2]

mask = (
    (grid_x > -3)
    & (grid_x < 3)
    & (grid_y > -3)
    & (grid_y < 3)
    & (grid_z > -0.5)
    & (grid_z < 4)
)

grid_x = np.where(mask, grid_x, np.nan)
grid_y = np.where(mask, grid_y, np.nan)
grid_z = np.where(mask, grid_z, np.nan)

ax1.plot_surface(
    grid_x,
    grid_y,
    grid_z,

    alpha=0.06,
    color='purple'
)

# =========================================================
# LABELS
# =========================================================

ax1.text(
    span_center[0] + 1.0,
    span_center[1] - 0.4,
    span_center[2] - 0.12,

    r'$\mathrm{span}(D)$',

    fontsize=14,
    color='dodgerblue',
    weight='bold'
)

# =========================================================
# PANEL (a) STYLING
# =========================================================

ax1.set_title(
    '(a) Discriminative Span And Linear Classifier',

    fontsize=16,
    pad=8
)

ax1.set_xlabel(
    'Embedding Dimension 1',
    fontsize=10,
    labelpad=4
)

ax1.set_ylabel(
    'Embedding Dimension 2',
    fontsize=10,
    labelpad=4
)

ax1.set_zlabel(
    'Embedding Dimension 3',
    fontsize=10,
    labelpad=4
)

ax1.view_init(
    elev=24,
    azim=-58
)

ax1.set_xlim(-2.0, 2.6)
ax1.set_ylim(-2.0, 2.6)
ax1.set_zlim(-0.5, 4.0)

ax1.set_box_aspect((1, 1, 1.05))

ax1.legend(
    loc='upper left',
    fontsize=8
)

# =========================================================
# =========================================================
# PANEL (b):
# PROJECTION-BASED DIAGNOSTIC
# =========================================================
# =========================================================

ax2 = fig.add_subplot(122)

ax2.set_xlim(-3.0, 5.2)
ax2.set_ylim(-1.0, 4.0)

ax2.axis('off')

# =========================================================
# LEFT SIDE:
# GEOMETRIC SCHEMATIC
# =========================================================

# ---------------------------------------------------------
# span(D) plane
# ---------------------------------------------------------

plane = plt.Rectangle(
    (-2.0, 0.0),
    3.8,
    1.3,

    color='dodgerblue',
    alpha=0.18
)

ax2.add_patch(plane)

# ---------------------------------------------------------
# Difference vectors
# ---------------------------------------------------------

for _ in range(8):

    start_x = np.random.uniform(-1.6, 1.0)
    start_y = np.random.uniform(0.35, 1.0)

    dx = np.random.uniform(0.4, 0.8)

    dy = np.random.uniform(
        -0.03,
        0.03
    )

    ax2.arrow(
        start_x,
        start_y,

        dx,
        dy,

        width=0.012,
        head_width=0.10,
        head_length=0.10,

        color='black',
        alpha=0.55,

        length_includes_head=True
    )

# =========================================================
# PROJECTION GEOMETRY
# =========================================================

origin = np.array([0.0, 0.65])

# Projection lies ON plane
projection = np.array([1.7, 0.65])

# Residual vector MUST be orthogonal to plane
# therefore purely vertical

w_tip = np.array([
    projection[0],
    2.9
])

# =========================================================
# DRAW PROJECTION VECTOR
# =========================================================

ax2.arrow(
    origin[0],
    origin[1],

    projection[0] - origin[0],
    projection[1] - origin[1],

    width=0.028,
    head_width=0.13,
    head_length=0.15,

    color='crimson',

    length_includes_head=True
)

# =========================================================
# DRAW TRUE VECTOR w
# =========================================================

ax2.arrow(
    origin[0],
    origin[1],

    w_tip[0] - origin[0],
    w_tip[1] - origin[1],

    width=0.032,
    head_width=0.15,
    head_length=0.17,

    color='purple',

    length_includes_head=True
)

# =========================================================
# DRAW ORTHOGONAL RESIDUAL
# =========================================================

ax2.plot(
    [projection[0], w_tip[0]],
    [projection[1], w_tip[1]],

    linestyle='--',
    linewidth=3,
    color='seagreen'
)

# =========================================================
# RIGHT ANGLE MARKER
# =========================================================

right_angle_x = projection[0]
right_angle_y = projection[1]

square_size = 0.18

ax2.plot(
    [
        right_angle_x,
        right_angle_x + square_size,
        right_angle_x + square_size,
        right_angle_x
    ],

    [
        right_angle_y,
        right_angle_y,
        right_angle_y + square_size,
        right_angle_y + square_size
    ],

    color='black',
    linewidth=1.4
)

# =========================================================
# LABELS
# =========================================================

ax2.text(
    -1.8,
    1.45,

    r'$\mathrm{span}(D)$',

    fontsize=16,
    color='dodgerblue'
)

ax2.text(
    w_tip[0] + 0.12,
    w_tip[1] + 0.05,

    r'$\mathbf{w}$',

    fontsize=18,
    color='purple'
)

ax2.text(
    projection[0] - 0.35,
    projection[1] - 0.38,

    r'$\Pi_{\mathrm{span}(D)}(\mathbf{w})$',

    fontsize=12,
    color='crimson'
)

# =========================================================
# RIGHT SIDE:
# FORMULA + DESCRIPTION
# =========================================================

formula = (
    r'$\mathrm{RPE}'
    r'='
    r'\frac{\|\mathbf{w}-\Pi_{\mathrm{span}(D)}(\mathbf{w})\|}'
    r'{\|\mathbf{w}\|}$'
)

ax2.text(
    3.35,
    2.45,

    formula,

    fontsize=23,
    ha='center'
)

description = (
    'Measures the fraction of the classifier\n'
    'discriminative direction unexplained\n'
    'by synthetic transformations.'
)

ax2.text(
    3.35,
    1.45,

    description,

    fontsize=11.5,
    ha='center',
    color='dimgray'
)

# =========================================================
# PANEL TITLE
# =========================================================

ax2.set_title(
    '(b) Orthogonal Projection-Based Diagnostic',

    fontsize=16,
    pad=10
)

# =========================================================
# GLOBAL TITLE
# =========================================================

# fig.suptitle(
#     'Discriminative Span Geometry and Projection-Based Diagnostic',

#     fontsize=20,
#     y=0.95
# )

# =========================================================
# FINAL LAYOUT
# =========================================================

plt.subplots_adjust(
    left=0.02,
    right=0.98,
    bottom=0.06,
    top=0.86,
    wspace=0.05
)

plt.tight_layout(
    rect=[0, 0, 1, 0.93]
)

plt.show()