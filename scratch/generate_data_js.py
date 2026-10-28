import json

with open("data/dashboard_data.json", "r", encoding="utf-8") as f:
    dashboard_data = json.load(f)

with open("data/model_trees.json", "r", encoding="utf-8") as f:
    model_data = json.load(f)

js_content = f"""// FreshTrack Data and Model Bundle (Generated from Jupyter Notebook)
window.FRESHTRACK_DATA = {json.dumps(dashboard_data, indent=None)};
window.FRESHTRACK_MODEL = {json.dumps(model_data, indent=None)};
"""

with open("data.js", "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"Generated data.js successfully ({len(js_content)} chars)")
