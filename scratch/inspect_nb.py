import json

with open("FreshTrack_Food_Waste_Analytics(main) (1).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}")
for i, cell in enumerate(nb.get("cells", [])):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    lines = source.split("\n")
    headers = [l for l in lines if l.strip().startswith("#")]
    first_lines = [l.strip() for l in lines if l.strip()][:3]
    print(f"--- Cell {i} ({cell_type}) ---")
    if headers:
        print(" Headers:", headers)
    else:
        print(" Preview:", first_lines)
