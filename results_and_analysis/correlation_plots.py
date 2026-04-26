import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd

# =========================
# LOAD DATA
# =========================
with open(r"C:\Users\97433\Knowing_the_difference\results_and_analysis\parsed_results.json", "r") as f:
    data = json.load(f)


def normalize(text):
    return text.lower().replace(" ", "_")


def get_avg_test_f1(dataset):
    return sum(m["Test F1"] for m in dataset["downstream_performance"]) / len(dataset["downstream_performance"])

def get_best_test_f1(dataset):
    return max(m["Test F1"] for m in dataset["downstream_performance"])

foundation_models = ["resnet18", "clip", "dinov2"]
solvers = ["least_squares", "ridge", "nnls", "l1"]


# =========================
# 1. PER-FOUNDATION ANALYSIS
# =========================
for foundation in foundation_models:

    plt.figure()
    rows = []

    for solver in solvers:
        x = []
        y = []

        for dataset_name, dataset in data.items():

            avg_f1 = get_best_test_f1(dataset)

            for row in dataset["diagnostic_metrics"]:
                emb = normalize(row["Embedding Model"])
                sol = normalize(row["Solver"])

                if emb == foundation and sol == solver:

                    rel_error = row["Explained Fraction"]

                    # Filter invalid / unstable values
                    if rel_error is None or rel_error > 10:
                        continue

                    x.append(rel_error)
                    y.append(avg_f1)

        if len(x) >= 3:
            pearson_corr, p_val = pearsonr(x, y)
            spearman_corr, _ = spearmanr(x, y)

            rows.append({
                "Solver": solver,
                "Pearson r": round(pearson_corr, 3),
                "p-value": round(p_val, 4),
                "Spearman ρ": round(spearman_corr, 3),
                "Num Points": len(x)
            })

            x_sorted, y_sorted = zip(*sorted(zip(x, y)))

            plt.plot(
                x_sorted,
                y_sorted,
                marker='o',
                linestyle='-',
                label=f"{solver} (r={pearson_corr:.2f}, p={p_val:.3f}, ρ={spearman_corr:.2f}, n={len(x)})"
            )

    plt.xlabel("Explaination Fraction")
    plt.ylabel("Best Test F1")
    plt.title(f"{foundation.upper()} Correlation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(rows).sort_values(by="Solver")

    print(f"\n=== {foundation.upper()} CORRELATION TABLE ===\n")
    print(df.to_string(index=False))

    df.to_csv(f"correlation_{foundation}.csv", index=False)
    print(f"\nSaved: correlation_{foundation}.csv\n")


# =========================
# 2. GLOBAL ANALYSIS (MAIN RESULT)
# =========================
plt.figure()
rows = []

for solver in solvers:
    x = []
    y = []

    for dataset_name, dataset in data.items():

        avg_f1 = get_avg_test_f1(dataset)

        for row in dataset["diagnostic_metrics"]:
            emb = normalize(row["Embedding Model"])
            sol = normalize(row["Solver"])

            if emb in foundation_models and sol == solver:

                rel_error = row["Relative Error"]

                if rel_error is None or rel_error > 10:
                    continue

                x.append(rel_error)
                y.append(avg_f1)

    if len(x) >= 5:
        pearson_corr, p_val = pearsonr(x, y)
        spearman_corr, _ = spearmanr(x, y)

        rows.append({
            "Solver": solver,
            "Pearson r": round(pearson_corr, 3),
            "p-value": round(p_val, 4),
            "Spearman ρ": round(spearman_corr, 3),
            "Num Points": len(x)
        })

        x_sorted, y_sorted = zip(*sorted(zip(x, y)))

        plt.plot(
            x_sorted,
            y_sorted,
            marker='o',
            linestyle='-',
            label=f"{solver} (r={pearson_corr:.2f}, p={p_val:.3f}, ρ={spearman_corr:.2f}, n={len(x)})"
        )

plt.xlabel("Relative Projection Error")
plt.ylabel("Average Test F1")
plt.title("GLOBAL Correlation (All Foundation Models)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

df = pd.DataFrame(rows).sort_values(by="Solver")

print("\n=== GLOBAL CORRELATION TABLE ===\n")
print(df.to_string(index=False))

df.to_csv("correlation_global.csv", index=False)

print("\nSaved: correlation_global.csv\n")