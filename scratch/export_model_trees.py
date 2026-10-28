import json
import numpy as np
import pandas as pd
from scratch.export_dashboard_data import full_model, numeric_features, categorical_features

# Extract preprocessor metadata
preprocessor = full_model.named_steps["preprocessor"]
cat_encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
cat_categories = [list(cats) for cats in cat_encoder.categories_]
rf = full_model.named_steps["regressor"]

# Export trees in compact format
forest_data = []
for est in rf.estimators_:
    t = est.tree_
    tree_dict = {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "thresh": [round(float(v), 5) for v in t.threshold],
        "values": [round(float(v[0][0]), 2) for v in t.value]
    }
    forest_data.append(tree_dict)

model_export = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "categories": cat_categories,
    "forest": forest_data
}

with open("scratch/model_trees.json", "w", encoding="utf-8") as f:
    json.dump(model_export, f)

print("Saved model_trees.json, size:", len(json.dumps(model_export)))

# Verify JS-style evaluation in Python
def evaluate_tree(tree, x):
    node = 0
    while tree["left"][node] != -1:
        feat = tree["feature"][node]
        if x[feat] <= tree["thresh"][node]:
            node = tree["left"][node]
        else:
            node = tree["right"][node]
    return tree["values"][node]

def predict_js(input_dict):
    # Transform input
    x = []
    for f in numeric_features:
        x.append(float(input_dict.get(f, 0)))
    
    # One-hot
    conf = input_dict.get("Confidence in estimate", "")
    for cat in cat_categories[0]:
        x.append(1.0 if conf == cat else 0.0)
    
    region = input_dict.get("Region", "")
    for cat in cat_categories[1]:
        x.append(1.0 if region == cat else 0.0)
    
    preds = [evaluate_tree(t, x) for t in forest_data]
    return np.mean(preds)

# Test with Australia
test_input = {
    "combined figures (kg/capita/year)": 133,
    "Household estimate (kg/capita/year)": 102,
    "Retail estimate (kg/capita/year)": 9,
    "Food service estimate (kg/capita/year)": 22,
    "Confidence in estimate": "High Confidence",
    "Region": "Australia and New Zealand"
}
py_sk_pred = full_model.predict(pd.DataFrame([test_input]))[0]
js_sim_pred = predict_js(test_input)

print(f"Scikit-learn predict: {py_sk_pred:.2f}")
print(f"JS simulator predict: {js_sim_pred:.2f}")
print(f"Difference: {abs(py_sk_pred - js_sim_pred):.4f}")
