import json

def records_to_markdown_table(records: list[dict]) -> str:
    if not records:
        return ""
    
    headers = list(records[0].keys())
    
    def fmt(v):
        if isinstance(v, float) and v > 1e8:
            exp = int(f"{v:.2e}".split("e")[1])
            mantissa = v / (10 ** exp)
            return f"{mantissa:.2f}e{exp:02d}"
        return str(v)
    
    rows = [[fmt(r.get(h, "")) for h in headers] for r in records]
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    
    def make_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |"
    
    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [make_row(headers), separator] + [make_row(r) for r in rows]
    return "\n".join(lines)


SECTION_MAP = {
    "downstream_performance": ("1", "Downstream Performance"),
    "diagnostic_metrics":     ("2", "Diagnostic Metrics (Embedding Space Analysis)"),
}

def unparse_to_markdown(parsed: dict) -> str:
    blocks = []
    for ds_idx, (dataset_name, sections) in enumerate(parsed.items(), start=1):
        blocks.append(f"### {ds_idx}. {dataset_name}\n")
        for section_key, (sub_num, section_title) in SECTION_MAP.items():
            if section_key not in sections:
                continue
            blocks.append(f"#### {ds_idx}.{sub_num} {section_title}\n")
            blocks.append(records_to_markdown_table(sections[section_key]))
            blocks.append("")
    return "\n".join(blocks)


with open(r"C:\Users\97433\Knowing_the_difference\results_and_analysis\parsed_results.json") as f:
    parsed = json.load(f)

markdown_output = unparse_to_markdown(parsed)

with open("reconstructed_results.md", "w") as f:
    f.write(markdown_output)