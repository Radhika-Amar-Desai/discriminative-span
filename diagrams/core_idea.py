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
# GENERATE DATA
# =========================================================

n_points = 14

# ---------------------------------------------------------
# Real Class A (blue stars)
# Compact cloud near origin
# ---------------------------------------------------------

x_a = np.random.normal(-0.5, 0.45, n_points)

y_a = np.random.normal(-0.8, 0.45, n_points)

z_a = np.random.normal(0.0, 0.08, n_points)

real_A = np.vstack([
    x_a,
    y_a,
    z_a
]).T

# ---------------------------------------------------------
# Synthetic Class B (orange circles)
#
# IMPORTANT:
#
# Synthetic transformations mostly span
# ONLY the x-y plane.
#
# Very weak z variation.
#
# So:
# span(D) ≈ planar.
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
# Real Class B (green circles)
#
# CRITICAL GEOMETRY:
#
# Real positives are NOT simply translated
# versions of synthetic_B.
#
# They contain a strong z component
# that synthetic transformations never span.
#
# Yet they remain on same semantic side.
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

# IMPORTANT:
# strong independent z direction

# z_b = np.random.normal(
#     2.8,
#     0.35,
#     n_points
# )

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
# CREATE FIGURE
# =========================================================

fig = plt.figure(figsize=(13, 11))

ax = fig.add_subplot(111, projection='3d')

# =========================================================
# PLOT POINTS
# =========================================================

ax.scatter(
    real_A[:, 0],
    real_A[:, 1],
    real_A[:, 2],
    marker='*',
    s=240,
    color='royalblue',
    label='Real Data (Class A)'
)

ax.scatter(
    synthetic_B[:, 0],
    synthetic_B[:, 1],
    synthetic_B[:, 2],
    marker='o',
    s=90,
    color='darkorange',
    edgecolors='black',
    label='Synthetic Data (Class B)'
)

ax.scatter(
    real_B[:, 0],
    real_B[:, 1],
    real_B[:, 2],
    marker='o',
    s=90,
    color='forestgreen',
    edgecolors='black',
    label='Real Data (Class B)'
)

# =========================================================
# DIFFERENCE VECTORS
# =========================================================

D = synthetic_B - real_A

# =========================================================
# DRAW DIFFERENCE VECTORS
# =========================================================

for i in range(n_points):

    start = real_A[i]

    direction = D[i]

    ax.quiver(
        start[0],
        start[1],
        start[2],

        direction[0],
        direction[1],
        direction[2],

        arrow_length_ratio=0.12,
        color='black',
        linewidth=1.5,
        alpha=0.9
    )

# =========================================================
# COMPUTE SPAN USING PCA
# =========================================================

pca = PCA(n_components=2)

pca.fit(D)

v1 = pca.components_[0]

v2 = pca.components_[1]

# span_center = np.mean(synthetic_B, axis=0)
span_center = (
    np.mean(synthetic_B, axis=0)
    - np.array([1.4, 1.3, 0.0])
)

# =========================================================
# CREATE SPAN PLANE
# =========================================================

plane_range = np.linspace(-2.0, 2.0, 12)

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

ax.plot_surface(
    xx,
    yy,
    zz,
    alpha=0.18,
    color='dodgerblue'
)

# =========================================================
# TRAIN LINEAR CLASSIFIER
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

# =========================================================
# CLASSIFIER NORMAL VECTOR
# =========================================================

w = clf.coef_[0]

w = w / np.linalg.norm(w)

b = clf.intercept_[0]

# =========================================================
# DRAW CLASSIFIER NORMAL VECTOR
# =========================================================

origin = np.mean(X, axis=0)

normal_scale = 2.5

ax.quiver(
    origin[0],
    origin[1],
    origin[2],

    w[0] * normal_scale,
    w[1] * normal_scale,
    w[2] * normal_scale,

    color='purple',
    linewidth=4,
    arrow_length_ratio=0.08
)

# =========================================================
# LABEL NORMAL VECTOR
# =========================================================

ax.text(
    origin[0] + w[0] * normal_scale,
    origin[1] + w[1] * normal_scale,
    origin[2] + w[2] * normal_scale + 0.2,
    r'$\mathbf{w}$',
    fontsize=24,
    color='purple',
    weight='bold'
)

# # =========================================================
# # DRAW CLASSIFIER HYPERPLANE
# # =========================================================

# grid_x, grid_y = np.meshgrid(
#     np.linspace(-1.8, 2.5, 10),
#     np.linspace(-1.5, 2.5, 10)
# )

# if abs(w[2]) > 1e-6:

#     grid_z = (
#         -w[0] * grid_x
#         -w[1] * grid_y
#         -b
#     ) / w[2]

#     # Keep only visible region
#     grid_z = np.where(
#         (grid_z > -0.5) & (grid_z < 4.0),
#         grid_z,
#         np.nan
#     )

#     ax.plot_surface(
#         grid_x,
#         grid_y,
#         grid_z,
#         alpha=0.12,
#         color='purple'
#     )

# =========================================================
# DRAW CLASSIFIER HYPERPLANE
# =========================================================

# ---------------------------------------------------------
# Point on hyperplane
# ---------------------------------------------------------

plane_center = -b * w

# ---------------------------------------------------------
# Push plane slightly backward
# along negative normal direction
# ---------------------------------------------------------

plane_center = (
    plane_center
    - 0.6 * w
)

# ---------------------------------------------------------
# Construct orthonormal basis
# for the hyperplane
# ---------------------------------------------------------

arb = np.array([1, 0, 0])

if np.abs(np.dot(arb, w)) > 0.9:
    arb = np.array([0, 1, 0])

u = np.cross(w, arb)
u = u / np.linalg.norm(u)

v = np.cross(w, u)
v = v / np.linalg.norm(v)

# ---------------------------------------------------------
# Local plane patch coordinates
# ---------------------------------------------------------

plane_size = 4.2

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

# ---------------------------------------------------------
# Keep only visible region
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# Plot hyperplane
# ---------------------------------------------------------

ax.plot_surface(
    grid_x,
    grid_y,
    grid_z,
    alpha=0.08,
    color='purple'
)

# =========================================================
# COMPUTE GEOMETRIC ALIGNMENT
# =========================================================

span_normal = np.cross(v1, v2)

span_normal = (
    span_normal
    / np.linalg.norm(span_normal)
)

cos_theta = np.abs(
    np.dot(w, span_normal)
)

angle_deg = np.degrees(
    np.arccos(
        np.clip(cos_theta, -1, 1)
    )
)

print(
    f"Angle between classifier normal "
    f"and span normal: {angle_deg:.2f}°"
)

# =========================================================
# ANNOTATIONS
# =========================================================

ax.text(
    span_center[0] + 1.4,
    span_center[1] - 0.4,
    span_center[2] - 0.15,
    "Span(D)",
    fontsize=15,
    color='dodgerblue',
    weight='bold'
)

ax.text(
    -1.2,
    0.3,
    2.5,
    "Classifier" + "\n" + "Hyperplane",
    fontsize=14,
    color='purple',
    weight='bold'
)

# Background text box improves readability

# ax.text(
#     -2.15,
#     -1.55,
#     1.05,

#     "Difference vectors span\nsynthetic transformation subspace",

#     fontsize=11,
#     color='black',

#     bbox=dict(
#         facecolor='white',
#         alpha=0.75,
#         edgecolor='none',
#         pad=4
#     )
# )

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
    'Embedding Space Geometry:\n'
    'Difference Vectors and Linear Classifier',
    fontsize=19,
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

ax.set_xlim(-2.5, 3.0)

ax.set_ylim(-2.5, 3.0)

ax.set_zlim(-0.5, 4.0)

# =========================================================
# GRID
# =========================================================

ax.grid(True)

# =========================================================
# LEGEND
# =========================================================

ax.legend(
    loc='upper left',
    fontsize=11
)

# =========================================================
# LAYOUT FIX
# =========================================================

fig.subplots_adjust(
    top=0.88,
    left=0.02,
    right=0.98
)

# =========================================================
# SHOW
# =========================================================

plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.show()