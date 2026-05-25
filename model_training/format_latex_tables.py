import pandas as pd


# =========================================================
# DATA
# =========================================================

data = [
    # -----------------------------------------------------
    # Pneumonia CXR
    # -----------------------------------------------------
    ["Pneumonia CXR", "ResNet-18",       1.0000, 1.0000, 0.6490, 0.6392],
    ["Pneumonia CXR", "MobileNet-V2",   1.0000, 1.0000, 0.5080, 0.4547],
    ["Pneumonia CXR", "EfficientNet-B0",1.0000, 1.0000, 0.6058, 0.5895],

    # -----------------------------------------------------
    # Skin Lesion
    # -----------------------------------------------------
    ["Skin Lesion", "ResNet-18",        1.0000, 1.0000, 0.9767, 0.9767],
    ["Skin Lesion", "MobileNet-V2",     1.0000, 1.0000, 0.9535, 0.9535],
    ["Skin Lesion", "EfficientNet-B0",  1.0000, 1.0000, 0.6744, 0.6977],

    # -----------------------------------------------------
    # Toy Watermark
    # -----------------------------------------------------
    ["Toy Watermark", "ResNet-18",      1.0000, 1.0000, 0.9889, 0.9889],
    ["Toy Watermark", "MobileNet-V2",   1.0000, 1.0000, 0.9667, 0.9668],
    ["Toy Watermark", "EfficientNet-B0",1.0000, 1.0000, 0.9556, 0.9557],

    # -----------------------------------------------------
    # Horses and Zebras
    # -----------------------------------------------------
    ["Horses and Zebras", "ResNet-18",      1.0000, 1.0000, 0.9750, 0.9751],
    ["Horses and Zebras", "MobileNet-V2",   1.0000, 1.0000, 0.9792, 0.9812],
    ["Horses and Zebras", "EfficientNet-B0",1.0000, 1.0000, 0.9542, 0.9543],

    # -----------------------------------------------------
    # Apples and Oranges
    # -----------------------------------------------------
    ["Apples and Oranges", "ResNet-18",      1.0000, 1.0000, 0.8785, 0.8649],
    ["Apples and Oranges", "MobileNet-V2",   1.0000, 1.0000, 0.8947, 0.8850],
    ["Apples and Oranges", "EfficientNet-B0",1.0000, 1.0000, 0.8603, 0.8571],
]


columns = [
    "Dataset",
    "Model",
    "Train Acc",
    "Train F1",
    "Test Acc",
    "Test F1"
]


df = pd.DataFrame(
    data,
    columns=columns
)


# =========================================================
# GENERATE LATEX TABLE
# =========================================================

latex = []

latex.append(r"\begin{table*}[t]")
latex.append(r"\centering")
latex.append(
    r"\caption{Combined downstream performance across all datasets.}"
)
latex.append(
    r"\label{tab:combined_downstream}"
)

latex.append(r"\begin{tabular}{llcccc}")
latex.append(r"\toprule")

latex.append(
    r"\textbf{Dataset} & "
    r"\textbf{Model} & "
    r"\textbf{Train Acc} & "
    r"\textbf{Train F1} & "
    r"\textbf{Test Acc} & "
    r"\textbf{Test F1} \\"
)

latex.append(r"\midrule")


# =========================================================
# MULTIROW GENERATION
# =========================================================

grouped = df.groupby("Dataset")

for dataset_name, group in grouped:

    group = group.reset_index(drop=True)

    n_rows = len(group)

    for idx, row in group.iterrows():

        if idx == 0:

            dataset_cell = (
                rf"\multirow{{{n_rows}}}{{*}}{{{dataset_name}}}"
            )

        else:

            dataset_cell = ""

        latex.append(
            f"{dataset_cell} & "
            f"{row['Model']} & "
            f"{row['Train Acc']:.4f} & "
            f"{row['Train F1']:.4f} & "
            f"{row['Test Acc']:.4f} & "
            f"{row['Test F1']:.4f} \\\\"
        )

    latex.append(r"\midrule")


# =========================================================
# FINALIZE
# =========================================================

latex[-1] = r"\bottomrule"

latex.append(r"\end{tabular}")
latex.append(r"\end{table*}")


# =========================================================
# PRINT
# =========================================================

latex_code = "\n".join(latex)

print(latex_code)


# =========================================================
# OPTIONAL SAVE
# =========================================================

with open("combined_downstream_table.tex", "w") as f:
    f.write(latex_code)

print(
    "\nSaved LaTeX table to "
    "'combined_downstream_table.tex'"
)