import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# =========================================================
# FIGURE STYLE
# =========================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
})

# =========================================================
# COLORS
# =========================================================
BG_COLOR = '#ffffff'

BLUE_FILL = '#e8f1fb'
BLUE_ACCENT = '#2f80ed'

PURPLE_FILL = '#f4ecff'
PURPLE_ACCENT = '#7b1fa2'

TEXT_DARK = '#222222'
TEXT_LIGHT = 'dimgray'

ARROW_COLOR = '#4a4a4a'
GRAY_COLOR = '#b6b6b6'

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(figsize=(18, 5))

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

ax.set_xlim(0, 21)
ax.set_ylim(0, 6)

ax.axis('off')

# =========================================================
# TITLE
# =========================================================

ax.text(
    10.5,
    5.55,

    "Low-Rank Approximation of Difference Vector Matrix",

    fontsize=23,
    ha='center',
    fontweight='bold',
    color=TEXT_DARK
)

# =========================================================
# STEP 1 — DIFFERENCE MATRIX D
# =========================================================

D_box = Rectangle(
    (0.8, 1.3),
    3.2,
    3.0,

    linewidth=2.5,
    edgecolor=BLUE_ACCENT,
    facecolor=BLUE_FILL
)

ax.add_patch(D_box)

# Matrix entries
matrix_entries = [
    [r'$d_{11}$', r'$d_{12}$', r'$\cdots$', r'$d_{1m}$'],
    [r'$d_{21}$', r'$d_{22}$', r'$\cdots$', r'$d_{2m}$'],
    [r'$\vdots$', r'$\vdots$', r'$\ddots$', r'$\vdots$'],
    [r'$d_{n1}$', r'$d_{n2}$', r'$\cdots$', r'$d_{nm}$'],
]

x_positions = [1.2, 1.9, 2.7, 3.4]
y_positions = [3.7, 3.0, 2.3, 1.6]

for i, y in enumerate(y_positions):

    for j, x in enumerate(x_positions):

        ax.text(
            x,
            y,

            matrix_entries[i][j],

            fontsize=15,
            ha='center',
            va='center',
            color=TEXT_DARK
        )

# Labels
ax.text(
    2.4,
    4.7,

    r'Difference Matrix $\mathbf{D}$',

    fontsize=18,
    ha='center'
)

ax.text(
    2.4,
    0.8,

    r'Rows correspond to difference vectors $(d_i)$',

    fontsize=11,
    ha='center',
    color=TEXT_LIGHT
)

# =========================================================
# ARROW 1
# =========================================================

arrow1 = FancyArrowPatch(
    (4.3, 2.8),
    (5.3, 2.8),

    arrowstyle='simple',
    mutation_scale=18,
    linewidth=1.5,
    color=ARROW_COLOR
)

ax.add_patch(arrow1)

# =========================================================
# STEP 2 — EFFECTIVE RANK ESTIMATION
# =========================================================

x_vals = [5.8, 6.1, 6.4, 6.7, 7.0, 7.3, 7.6, 7.9, 8.2]
y_vals = [4.0, 3.6, 3.2, 2.7, 2.2, 1.8, 1.5, 1.3, 1.2]

# Singular value decay curve
ax.plot(
    x_vals,
    y_vals,
    color='#f59e0b',   # Orange line
    linewidth=1.5
)

# Singular value points
ax.scatter(
    x_vals,
    y_vals,
    color='#f59e0b',   # Orange line
    # color=TEXT_DARK,
    s=32
)

# Effective rank cutoff
cutoff_x = 7.3

ax.plot(
    [cutoff_x, cutoff_x],
    [1.0, 4.2],
    
    linestyle='--',
    linewidth=2,
    color=GRAY_COLOR
)

# Retained region
ax.fill_between(
    x_vals[:6],
    y_vals[:6],
    1.0,

    color='#cfe2ff',
    alpha=0.9
)

# Labels
ax.text(
    7.0,
    4.7,

    'Effective Rank Estimation',

    fontsize=18,
    ha='center'
)

ax.text(
    cutoff_x + 0.08,
    0.82,

    r'$k$',

    fontsize=15
)

ax.text(
    7.0,
    0.45,

    'Selecting dominant singular directions',

    fontsize=11,
    ha='center',
    color=TEXT_LIGHT
)

# =========================================================
# ARROW 2
# =========================================================

arrow2 = FancyArrowPatch(
    (8.6, 2.8),
    (9.6, 2.8),

    arrowstyle='simple',
    mutation_scale=18,
    linewidth=1.5,
    color=ARROW_COLOR
)

ax.add_patch(arrow2)

# =========================================================
# STEP 3 — TRUNCATED SVD
# =========================================================

svd_y = 2.15

block_x = [10.0, 11.8, 13.6]

labels = [
    r'$\mathbf{U_k}$',
    r'$\mathbf{\Sigma_k}$',
    r'$\mathbf{V_k^T}$'
]

# Draw SVD blocks
for x, label in zip(block_x, labels):

    rect = Rectangle(
        (x, svd_y),
        1.0,
        1.2,

        linewidth=2,
        edgecolor=PURPLE_ACCENT,
        facecolor=PURPLE_FILL
    )

    ax.add_patch(rect)

    ax.text(
        x + 0.5,
        svd_y + 0.6,

        label,

        fontsize=16,
        ha='center',
        va='center',
        color=PURPLE_ACCENT
    )

# Multiplication symbols
ax.text(
    11.35,
    2.75,

    r'$\times$',

    fontsize=18,
    ha='center',
    va='center',
    color=TEXT_DARK
)

ax.text(
    13.15,
    2.75,

    r'$\times$',

    fontsize=18,
    ha='center',
    va='center',
    color=TEXT_DARK
)

# Labels
ax.text(
    12.3,
    4.7,

    'Truncated SVD',

    fontsize=18,
    ha='center'
)

ax.text(
    12.3,
    1.25,

    r'$\mathbf{D_k = U_k \Sigma_k V_k^T}$',

    fontsize=18,
    ha='center',
    color=PURPLE_ACCENT
)

# =========================================================
# ARROW 3
# =========================================================

arrow3 = FancyArrowPatch(
    (14.9, 2.8),
    (15.9, 2.8),

    arrowstyle='simple',
    mutation_scale=18,
    linewidth=1.5,
    color=ARROW_COLOR
)

ax.add_patch(arrow3)

# =========================================================
# STEP 4 — REDUCED MATRIX D_k
# =========================================================

reduced_entries = [
    [r'$d_{11}$', r'$d_{12}$', r'$\cdots$', r'$d_{1n}$'],
    [r'$d_{21}$', r'$d_{22}$', r'$\cdots$', r'$d_{2n}$'],
    [r'$\vdots$', r'$\vdots$', r'$\ddots$', r'$\vdots$'],
    [r'$d_{k1}$', r'$d_{k2}$', r'$\cdots$', r'$d_{kn}$'],
]

# Matrix positions
x_positions = [16.6, 17.5, 18.5, 19.6]
y_positions = [3.7, 3.0, 2.3, 1.6]

# Draw entries
for i, y in enumerate(y_positions):

    for j, x in enumerate(x_positions):

        ax.text(
            x,
            y,

            reduced_entries[i][j],

            fontsize=15,
            ha='center',
            va='center',
            color=TEXT_DARK
        )

# =========================================================
# MATRIX BRACKETS
# =========================================================

# LEFT BRACKET
ax.plot(
    [16.0, 16.0],
    [1.3, 4.1],

    color=BLUE_ACCENT,
    linewidth=2.4
)

ax.plot(
    [16.0, 16.45],
    [4.1, 4.1],

    color=BLUE_ACCENT,
    linewidth=2.4
)

ax.plot(
    [16.0, 16.45],
    [1.3, 1.3],

    color=BLUE_ACCENT,
    linewidth=2.4
)

# RIGHT BRACKET
ax.plot(
    [20.1, 20.1],
    [1.3, 4.1],

    color=BLUE_ACCENT,
    linewidth=2.4
)

ax.plot(
    [19.65, 20.1],
    [4.1, 4.1],

    color=BLUE_ACCENT,
    linewidth=2.4
)

ax.plot(
    [19.65, 20.1],
    [1.3, 1.3],

    color=BLUE_ACCENT,
    linewidth=2.4
)

# =========================================================
# LABELS
# =========================================================

ax.text(
    18.0,
    4.8,

    r'Reduced Matrix $\mathbf{D_k}$',

    fontsize=18,
    ha='center'
)

ax.text(
    18.0,
    0.55,

    'Low-dimensional structured\ntransformation subspace',

    fontsize=10.5,
    ha='center',
    color=TEXT_LIGHT
)

# =========================================================
# SHOW
# =========================================================

plt.tight_layout()
plt.show()