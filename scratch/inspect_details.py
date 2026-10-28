import json

with open("FreshTrack_Food_Waste_Analytics(main) (1).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get("cells", [])):
    print(f"\n==================== CELL {i} ({cell.get('cell_type')}) ====================")
    print("".join(cell.get("source", [])))
    outputs = cell.get("outputs", [])
    if outputs:
        print("--- Outputs ---")
        for out in outputs:
            if out.get("output_type") == "stream":
                print("".join(out.get("text", []))[:600])
            elif "data" in out:
                if "text/plain" in out["data"]:
                    print("".join(out["data"]["text/plain"])[:600])
