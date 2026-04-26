import re
import json

from markdown_results import MARKDOWN_RESULTS

def is_separator_row(row):
    """A separator row has only dashes, pipes, colons, and spaces."""
    return bool(re.match(r"^\s*\|[\s\-:|]+\|\s*$", row))

def parse_table(lines):
    """Parse a markdown table into a list of dicts."""
    # Keep only pipe-starting lines, skip blanks
    rows = [l for l in lines if l.strip().startswith("|")]
    if len(rows) < 2:
        return []

    # Find the header row: the first non-separator row
    header_idx = next((i for i, r in enumerate(rows) if not is_separator_row(r)), None)
    if header_idx is None:
        return []

    headers = [h.strip() for h in rows[header_idx].strip().strip("|").split("|")]

    # Data rows: everything after the header, skipping separator rows
    data_rows = [r for r in rows[header_idx + 1:] if not is_separator_row(r)]

    records = []
    for row in data_rows:
        values = [v.strip() for v in row.strip().strip("|").split("|")]
        if len(values) != len(headers):
            continue
        record = {}
        for h, v in zip(headers, values):
            # Attempt numeric coercion
            try:
                record[h] = int(v)
            except ValueError:
                try:
                    record[h] = float(v)
                except ValueError:
                    record[h] = v
        records.append(record)
    return records


def parse_markdown(md: str) -> dict:
    result = {}
    current_dataset = None
    current_section = None
    table_lines = []

    section_map = {
        "1": "downstream_performance",
        "2": "diagnostic_metrics",
    }

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Match top-level dataset heading: ### N. Name
        ds_match = re.match(r"^###\s+\d+\.\s+(.+)", line)
        if ds_match:
            # Flush any pending table
            if current_dataset and current_section and table_lines:
                result[current_dataset][current_section] = parse_table(table_lines)
                table_lines = []
            current_dataset = ds_match.group(1).strip()
            result[current_dataset] = {}
            current_section = None
            i += 1
            continue

        # Match sub-section heading: #### N.M Title
        sub_match = re.match(r"^####\s+\d+\.(\d+)\s+(.+)", line)
        if sub_match:
            # Flush pending table
            if current_dataset and current_section and table_lines:
                result[current_dataset][current_section] = parse_table(table_lines)
                table_lines = []
            sub_num = sub_match.group(1)
            current_section = section_map.get(sub_num, sub_match.group(2).strip().lower().replace(" ", "_"))
            i += 1
            continue

        # Accumulate table lines
        if line.strip().startswith("|") and current_dataset and current_section:
            table_lines.append(line)
        elif table_lines:
            # Non-table line after table content — flush
            result[current_dataset][current_section] = parse_table(table_lines)
            table_lines = []

        i += 1

    # Flush any remaining table
    if current_dataset and current_section and table_lines:
        result[current_dataset][current_section] = parse_table(table_lines)

    return result


parsed = parse_markdown(MARKDOWN_RESULTS)

output_path = "parsed_results.json"
with open(output_path, "w") as f:
    json.dump(parsed, f, indent=2)

print(f"Saved to {output_path}")
print(f"Datasets parsed: {list(parsed.keys())}")
for ds, sections in parsed.items():
    for sec, rows in sections.items():
        print(f"  {ds} → {sec}: {len(rows)} rows")