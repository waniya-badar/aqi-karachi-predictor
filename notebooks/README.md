# EDA Notebook Guide

## Running the EDA Notebook

The `eda_and_explainability.ipynb` notebook contains all exploratory data analysis with dynamic visualizations.

### Method 1: Using Jupyter Notebook (Recommended)

```bash
# Navigate to project root
cd d:\Pycharm\Summer\AQIPredictorKarachi\aqi-predictor-karachi

# Launch Jupyter Notebook
jupyter notebook

# In the browser, navigate to:
# notebooks/eda_and_explainability.ipynb
```

### Method 2: Using Jupyter Lab

```bash
# Navigate to project root
cd d:\Pycharm\Summer\AQIPredictorKarachi\aqi-predictor-karachi

# Launch Jupyter Lab
jupyter lab

# Open notebooks/eda_and_explainability.ipynb from the file browser
```

### Method 3: Using VS Code

1. Open the project folder in VS Code
2. Navigate to `notebooks/eda_and_explainability.ipynb`
3. Click on the file to open it
4. Select your Python kernel (the venv interpreter)
5. Click "Run All" or run cells individually

### Method 4: Using PyCharm

1. Open the project in PyCharm
2. Navigate to `notebooks/eda_and_explainability.ipynb`
3. PyCharm will open it in the notebook editor
4. Select the project interpreter
5. Run cells using Shift+Enter

---

## What the Notebook Does

The notebook performs:

1. **Data Loading** - Fetches data from MongoDB Cloud
2. **Descriptive Statistics** - Summary statistics of features
3. **Distribution Plots** - Histograms, box plots
4. **Correlation Analysis** - Heatmaps showing feature relationships
5. **Time Series Plots** - AQI trends over time
6. **Feature Importance** - If models are loaded
7. **SHAP/LIME Analysis** - Model explainability (optional)

All plots are generated **dynamically** from your latest MongoDB data!

---

## Prerequisites

Ensure you have Jupyter installed:

```bash
pip install jupyter notebook ipykernel
```

Or if using your venv:

```bash
.\venv\Scripts\activate
pip install jupyter notebook ipykernel
```

---

## Output

- Visualizations appear inline in the notebook
- Plots are saved to `notebooks/plots/` directory
- You can export the notebook with outputs as HTML or PDF

---

## Tips

- Run cells sequentially (top to bottom)
- Modify visualizations as needed
- Re-run anytime to get latest data analysis
- Use `Kernel → Restart & Run All` for fresh analysis

---

## Environment Variables

Make sure your `.env` file in the project root contains:

```
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB_NAME=aqi_karachi
```

The notebook will automatically load these to connect to MongoDB.
