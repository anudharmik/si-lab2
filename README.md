# FreshTrack — Food Waste Intelligence & Analytics Dashboard

A modern, executive dashboard frontend built from the [FreshTrack Food Waste Analytics & Prediction](file:///e:/si-lab-kunte/FreshTrack_Food_Waste_Analytics(main)%20(1).ipynb) Jupyter notebook.

---

## Highlights

- **Authentic UNEP Data**: Synthesized directly from the 214 country records in the official UN Environment Programme Food Waste Index.
- **Strictly Segregated Sections**:
  1. **Global Pulse & Executive Overview**: Top-line KPI metrics (931.3M tonnes annual waste, 61.2% household share, 126.8 kg/capita global average, 214 nations).
  2. **Interactive Filters**: Dynamic filtering by region, estimate confidence, and quick search. Real-time recalculation of all metrics.
  3. **Country Rankings & Explorer**: Interactive horizontal bar rankings with metric toggles (`Per Capita`, `Total Tonnes`, `Household kg`) and a searchable country browser.
  4. **Waste Sources Comparison**: Donut chart and per-capita bar comparisons analyzing Household vs. Food Service vs. Retail discard.
  5. **Regional Benchmark**: 17 global regions ranked by per-capita waste alongside summary statistics.
  6. **Interactive ML Estimator**: Real-time client-side execution of the notebook's 200-tree Random Forest Regressor with preset scenarios, custom sliders, and actual-vs-predicted test comparison.
  7. **Full Data Registry & Export**: Complete 214-country table with sorting, pagination, search, and one-click CSV and JSON export.
  8. **Country Deep-Dive Drawer**: Slide-out profile sheet with source breakdowns, global comparison benchmarks, and direct links to official UNEP country reports.
- **Premium Aesthetics**:
  - Executive dark obsidian & emerald slate theme with light mode toggle.
  - Distinctive typography using Google Fonts (**Outfit** for headings & numbers + **Plus Jakarta Sans** for body & controls).
  - Strict tabular numeral alignment (`font-variant-numeric: tabular-nums`).
  - Subtle micro-interactions, responsive charts, glassmorphism cards.
- **No AI Slop / No Unnecessary Versions**:
  - Clean, plain-English explanations.
  - Zero meaningless buzzwords or bogus version identifiers.

---

## How to Run Locally

### Option 1: Standalone Browser (Zero Installation)
Simply double-click or open [index.html](file:///e:/si-lab-kunte/index.html) in any modern web browser (Chrome, Edge, Firefox, Safari).

### Option 2: Using the Included Python Server
Run the lightweight zero-dependency server:
```bash
python server.py
```
Then navigate to:
```
http://localhost:8000/index.html
```

---

## Project Structure

```
si-lab-kunte/
├── index.html                                        # Semantic, accessible dashboard structure
├── style.css                                         # Design system, CSS tokens, dark/light themes, typography
├── app.js                                            # Dashboard controller, filter engine, Chart.js, ML predictor
├── data.js                                           # Self-contained data & model bundle for zero-CORS offline execution
├── server.py                                         # Zero-dependency local development server
├── assets/
│   └── chart.umd.min.js                              # Local offline Chart.js v4 bundle
├── data/
│   ├── dashboard_data.json                           # 214-country records, regional stats & test metrics
│   └── model_trees.json                              # Serialized 200-tree Random Forest ensemble
├── FreshTrack_Food_Waste_Analytics(main) (1).ipynb   # Original analysis & modeling notebook
└── food_waste_data.csv                               # UNEP Food Waste Index country dataset
```
