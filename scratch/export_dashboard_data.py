import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("food_waste_data.csv")
df.columns = df.columns.str.strip()

# Numeric conversions
numeric_cols = [
    "combined figures (kg/capita/year)",
    "Household estimate (kg/capita/year)",
    "Household estimate (tonnes/year)",
    "Retail estimate (kg/capita/year)",
    "Retail estimate (tonnes/year)",
    "Food service estimate (kg/capita/year)",
    "Food service estimate (tonnes/year)"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Train ML model exactly as in notebook
TARGET = "Household estimate (tonnes/year)"
feature_cols = [
    "combined figures (kg/capita/year)",
    "Household estimate (kg/capita/year)",
    "Retail estimate (kg/capita/year)",
    "Food service estimate (kg/capita/year)",
    "Confidence in estimate",
    "Region"
]

model_df = df[feature_cols + [TARGET, "Country"]].dropna(subset=[TARGET]).copy()
X = model_df[feature_cols]
y = model_df[TARGET]

numeric_features = [
    "combined figures (kg/capita/year)",
    "Household estimate (kg/capita/year)",
    "Retail estimate (kg/capita/year)",
    "Food service estimate (kg/capita/year)"
]
categorical_features = ["Confidence in estimate", "Region"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

rf = RandomForestRegressor(n_estimators=200, random_state=42)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", rf)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
preds_test = model.predict(X_test)

mae = mean_absolute_error(y_test, preds_test)
rmse = mean_squared_error(y_test, preds_test) ** 0.5
r2 = r2_score(y_test, preds_test)

print(f"Trained model: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

# Also fit full model for live interactive predictions
full_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])
full_model.fit(X, y)

# Let's inspect test comparison
test_comparison = []
test_indices = y_test.index
for idx, pred in zip(test_indices, preds_test):
    c_name = model_df.loc[idx, "Country"]
    act = float(y_test.loc[idx])
    test_comparison.append({
        "country": c_name,
        "region": str(model_df.loc[idx, "Region"]),
        "actual": round(act),
        "predicted": round(float(pred)),
        "diff": round(float(pred) - act)
    })

# Format country records
countries_data = []
for _, row in df.iterrows():
    c_name = str(row["Country"]).strip()
    hh_tonnes = float(row["Household estimate (tonnes/year)"])
    ret_tonnes = float(row["Retail estimate (tonnes/year)"])
    fs_tonnes = float(row["Food service estimate (tonnes/year)"])
    total_tonnes = hh_tonnes + ret_tonnes + fs_tonnes

    hh_kg = float(row["Household estimate (kg/capita/year)"])
    ret_kg = float(row["Retail estimate (kg/capita/year)"])
    fs_kg = float(row["Food service estimate (kg/capita/year)"])
    comb_kg = float(row["combined figures (kg/capita/year)"])

    # Inferred population
    pop = round((hh_tonnes * 1000) / hh_kg) if hh_kg > 0 else 0

    countries_data.append({
        "name": c_name,
        "region": str(row["Region"]).strip(),
        "confidence": str(row["Confidence in estimate"]).strip(),
        "m49": int(row["M49 code"]) if pd.notnull(row["M49 code"]) else 0,
        "source": str(row["Source"]).strip(),
        "combined_kg": comb_kg,
        "household_kg": hh_kg,
        "household_tonnes": hh_tonnes,
        "retail_kg": ret_kg,
        "retail_tonnes": ret_tonnes,
        "foodservice_kg": fs_kg,
        "foodservice_tonnes": fs_tonnes,
        "total_tonnes": total_tonnes,
        "pop_est": pop
    })

# Regional aggregation
region_groups = df.groupby("Region")
regional_data = []
for region_name, group in region_groups:
    regional_data.append({
        "region": region_name,
        "country_count": len(group),
        "avg_combined_kg": round(float(group["combined figures (kg/capita/year)"].mean()), 2),
        "avg_household_kg": round(float(group["Household estimate (kg/capita/year)"].mean()), 2),
        "avg_retail_kg": round(float(group["Retail estimate (kg/capita/year)"].mean()), 2),
        "avg_foodservice_kg": round(float(group["Food service estimate (kg/capita/year)"].mean()), 2),
        "total_household_tonnes": round(float(group["Household estimate (tonnes/year)"].sum())),
        "total_retail_tonnes": round(float(group["Retail estimate (tonnes/year)"].sum())),
        "total_foodservice_tonnes": round(float(group["Food service estimate (tonnes/year)"].sum())),
        "total_waste_tonnes": round(float(group["Household estimate (tonnes/year)"].sum() +
                                          group["Retail estimate (tonnes/year)"].sum() +
                                          group["Food service estimate (tonnes/year)"].sum()))
    })

regional_data.sort(key=lambda r: r["avg_combined_kg"], reverse=True)

# Global source breakdown
total_hh_tonnes = float(df["Household estimate (tonnes/year)"].sum())
total_ret_tonnes = float(df["Retail estimate (tonnes/year)"].sum())
total_fs_tonnes = float(df["Food service estimate (tonnes/year)"].sum())
total_all_tonnes = total_hh_tonnes + total_ret_tonnes + total_fs_tonnes

sources_summary = {
    "total_tonnes": {
        "household": total_hh_tonnes,
        "retail": total_ret_tonnes,
        "foodservice": total_fs_tonnes,
        "all": total_all_tonnes
    },
    "shares_percent": {
        "household": round((total_hh_tonnes / total_all_tonnes) * 100, 1),
        "retail": round((total_ret_tonnes / total_all_tonnes) * 100, 1),
        "foodservice": round((total_fs_tonnes / total_all_tonnes) * 100, 1)
    },
    "averages_kg": {
        "household": round(float(df["Household estimate (kg/capita/year)"].mean()), 2),
        "retail": round(float(df["Retail estimate (kg/capita/year)"].mean()), 2),
        "foodservice": round(float(df["Food service estimate (kg/capita/year)"].mean()), 2),
        "combined": round(float(df["combined figures (kg/capita/year)"].mean()), 2)
    }
}

# How can we run model predictions in JS seamlessly?
# Let's inspect feature importances and tree or test grid to see if we can export a compact predictor:
# Random forest regressor feature importances:
enc = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
cat_feature_names = list(enc.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + cat_feature_names
importances = rf.feature_importances_
feature_imp_list = [
    {"feature": name, "importance": round(float(imp), 4)}
    for name, imp in zip(all_feature_names, importances)
]
feature_imp_list.sort(key=lambda x: x["importance"], reverse=True)

print("Top 10 Feature Importances:")
for f in feature_imp_list[:10]:
    print(f"  {f['feature']}: {f['importance']}")

output_payload = {
    "metrics": {
        "total_countries": len(df),
        "total_regions": len(regional_data),
        "global_total_tonnes": total_all_tonnes,
        "global_household_tonnes": total_hh_tonnes,
        "global_retail_tonnes": total_ret_tonnes,
        "global_foodservice_tonnes": total_fs_tonnes,
        "avg_combined_kg": sources_summary["averages_kg"]["combined"],
        "avg_household_kg": sources_summary["averages_kg"]["household"],
        "avg_retail_kg": sources_summary["averages_kg"]["retail"],
        "avg_foodservice_kg": sources_summary["averages_kg"]["foodservice"],
        "ml_mae": mae,
        "ml_rmse": rmse,
        "ml_r2": r2,
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    },
    "sources": sources_summary,
    "feature_importances": feature_imp_list[:12],
    "test_comparison": test_comparison,
    "regions": regional_data,
    "countries": countries_data
}

with open("scratch/dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

print("Saved scratch/dashboard_data.json successfully!")
