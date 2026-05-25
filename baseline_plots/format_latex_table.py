import pandas as pd

# =========================
# LOAD CSV FILES
# =========================

embedding_df = pd.read_csv("embedding_similarity_all_values.csv")
fid_df = pd.read_csv("fid_all_values.csv")
probe_df = pd.read_csv("linear_probe_accuracy_all_values.csv")

# =========================
# CLEAN COLUMN NAMES
# =========================

embedding_df.columns = embedding_df.columns.str.strip()
fid_df.columns = fid_df.columns.str.strip()
probe_df.columns = probe_df.columns.str.strip()

# =========================
# SELECT REQUIRED COLUMNS
# =========================

embedding_df = embedding_df[
    [
        "Foundation Model",
        "Dataset",
        "Metric Value",
        "Best Test F1"
    ]
].rename(columns={
    "Foundation Model": "Foundation",
    "Metric Value": "Embedding Sim"
})

fid_df = fid_df[
    [
        "Foundation Model",
        "Dataset",
        "Metric Value"
    ]
].rename(columns={
    "Foundation Model": "Foundation",
    "Metric Value": "FID"
})

probe_df = probe_df[
    [
        "Foundation Model",
        "Dataset",
        "Metric Value"
    ]
].rename(columns={
    "Foundation Model": "Foundation",
    "Metric Value": "Probe Acc"
})

# =========================
# MERGE TABLES
# =========================

merged_df = embedding_df.merge(
    fid_df,
    on=["Foundation", "Dataset"]
)

merged_df = merged_df.merge(
    probe_df,
    on=["Foundation", "Dataset"]
)

# =========================
# ROUND VALUES
# =========================

for col in ["FID", "Embedding Sim", "Probe Acc", "Best Test F1"]:
    merged_df[col] = merged_df[col].round(4)

# =========================
# SORT
# =========================

merged_df = merged_df.sort_values(
    by=["Foundation", "Dataset"]
)

# =========================
# GENERATE LATEX TABLE
# =========================

latex = []

latex.append(r"\begin{table*}[t]")
latex.append(r"\centering")
latex.append(r"\caption{Baseline evaluation metrics across datasets and embedding spaces.}")
latex.append(r"\label{tab:baseline_metrics}")
latex.append(r"\begin{tabular}{llcccc}")
latex.append(r"\hline")
latex.append(
    r"Foundation & Dataset & FID & Embedding Sim & Probe Acc & Best Test F1 \\"
)
latex.append(r"\hline")

# =========================
# MULTIROW GENERATION
# =========================

foundations = merged_df["Foundation"].unique()

for foundation in foundations:

    sub_df = merged_df[
        merged_df["Foundation"] == foundation
    ]

    n_rows = len(sub_df)

    for idx, (_, row) in enumerate(sub_df.iterrows()):

        foundation_cell = ""

        if idx == 0:
            foundation_cell = rf"\multirow{{{n_rows}}}{{*}}{{{foundation}}}"

        latex.append(
            f"{foundation_cell} & "
            f"{row['Dataset']} & "
            f"{row['FID']} & "
            f"{row['Embedding Sim']} & "
            f"{row['Probe Acc']} & "
            f"{row['Best Test F1']} \\\\"
        )

    latex.append(r"\hline")

latex.append(r"\end{tabular}")
latex.append(r"\end{table*}")

# =========================
# PRINT LATEX
# =========================

print("\n".join(latex))