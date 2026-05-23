import csv
import argparse

SOLVER_MAP = {
    "least_squares": "Least Squares",
    "ridge": "Ridge",
    "nnls": "NNLS",
    "l1": "L1"
}


def format_number(x, approx_threshold=1e-6):
    x = float(x)

    if abs(x) < approx_threshold:
        return r"$\approx 0$\textsuperscript{$\dagger$}"

    if abs(x - 1.0) < approx_threshold:
        return r"$\approx 1$"

    return f"{x:.3f}"


def convert_csv_to_rows(input_csv):
    rows = []

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["rank_method"] != "effective_rank":
                continue

            embedding = row["model"]
            solver = SOLVER_MAP[row["solver"]]
            k = int(float(row["k"]))
            rel_error = format_number(row["relative_error"])
            expl = format_number(row["explained_fraction"])
            pairs = int(row["num_pairs"])
            dim = int(row["embedding_dim"])

            latex_row = (
                f"{embedding} & {solver} & {k} & "
                f"{rel_error} & {expl} & {pairs} & {dim} \\\\"
            )

            rows.append(latex_row)

    return rows


def generate_latex(rows, caption, label):
    header = f"""\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{llccccc}}
\\toprule
\\textbf{{Embedding}} & \\textbf{{Solver}} & \\textbf{{Eff.\\ Rank}} & \\textbf{{Rel.\\ Error}} & \\textbf{{Expl.\\ Fraction}} & \\textbf{{Pairs}} & \\textbf{{Dim}} \\\\
\\midrule
"""

    footer = """
\\bottomrule
\\end{tabular}
\\end{table}
"""

    return header + "\n".join(rows) + footer


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to LaTeX table (.txt)")

    parser.add_argument("--input_csv", required=True, help="Path to input CSV")
    parser.add_argument("--output_txt", required=True, help="Path to output .txt file")
    parser.add_argument("--caption", default="Embedding space diagnostics.", help="Table caption")
    parser.add_argument("--label", default="tab:diag", help="LaTeX label")

    args = parser.parse_args()

    rows = convert_csv_to_rows(args.input_csv)
    latex = generate_latex(rows, args.caption, args.label)

    with open(args.output_txt, "w") as f:
        f.write(latex)

    print(f"LaTeX table written to: {args.output_txt}")


if __name__ == "__main__":
    main()