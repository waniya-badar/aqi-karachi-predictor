# How to Run EDA Analysis

## Quick Start

### Option 1: Jupyter Notebook (Easiest)

```bash
# 1. Navigate to project directory
cd d:\Pycharm\Summer\AQIPredictorKarachi\aqi-predictor-karachi

# 2. Activate virtual environment (if using one)
.\venv\Scripts\activate

# 3. Install Jupyter (if not already installed)
pip install jupyter notebook ipykernel matplotlib seaborn

# 4. Launch Jupyter Notebook
jupyter notebook

# 5. In the browser that opens, click:
#    notebooks → eda_and_explainability.ipynb

# 6. Run all cells:
#    Menu: Cell → Run All
#    Or: Shift+Enter to run each cell
```

### Option 2: VS Code

```bash
# 1. Open VS Code
# 2. Install Python and Jupyter extensions
# 3. Open project folder
# 4. Navigate to notebooks/eda_and_explainability.ipynb
# 5. Click "Select Kernel" → Choose your Python interpreter
# 6. Click "Run All" button at the top
```

### Option 3: PyCharm Professional

```bash
# 1. Open project in PyCharm
# 2. Navigate to notebooks/eda_and_explainability.ipynb
# 3. PyCharm automatically opens it in notebook editor
# 4. Select project interpreter if prompted
# 5. Click green play button to run cells
# 6. Or use Shift+Enter to run individual cells
```

---

## What You'll See

The notebook generates:

✅ **Summary Statistics** - Mean, median, std of all features  
✅ **Distribution Plots** - Histograms showing data distributions  
✅ **Correlation Heatmap** - Feature relationships  
✅ **Time Series Plots** - AQI trends over time  
✅ **Box Plots** - Outlier detection  
✅ **Feature Importance** - Which features matter most  
✅ **SHAP Analysis** (optional) - Model explainability  

All visualizations are **dynamic** and generated from your latest MongoDB data!

---

## Saved Outputs

Plots are automatically saved to: `notebooks/plots/`

You can find:
- `correlation_heatmap.png`
- `aqi_distribution.png`
- `feature_importance.png`
- And more...

---

## Troubleshooting

### Issue: "jupyter: command not found"

**Solution**: Install Jupyter
```bash
pip install jupyter notebook
```

### Issue: "No module named 'pymongo'"

**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Issue: Can't connect to MongoDB

**Solution**: Check `.env` file contains:
```
MONGODB_URI=mongodb+srv://...
MONGODB_DB_NAME=aqi_karachi
```

### Issue: Plots not showing

**Solution**: Add this at the top of notebook
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

---

## When to Run EDA

Run the notebook when you want to:

- ✅ Understand data patterns and trends
- ✅ Check for outliers or anomalies
- ✅ See feature correlations
- ✅ Analyze model performance
- ✅ Generate reports with visualizations
- ✅ Debug data quality issues

You can run it as often as you like - it's **not automated** by workflows, giving you full control.

---

## Tips

1. **Run Sequentially**: Execute cells from top to bottom
2. **Restart Kernel**: Use "Kernel → Restart & Run All" for clean slate
3. **Modify Freely**: Change visualizations, add new analyses
4. **Export Results**: File → Download as → HTML/PDF
5. **Fresh Data**: Re-run anytime to analyze latest MongoDB data

---

## Next Steps After EDA

After analyzing your data:

1. **Update Features**: Modify `src/feature_engineering.py` if needed
2. **Retrain Models**: Run `python pipelines/training_pipeline.py`
3. **Check Performance**: Review training_history in MongoDB
4. **Deploy Changes**: Push to GitHub for automated workflows

---

## Need Help?

- Check [QUICK_REFERENCE.md](../QUICK_REFERENCE.md) for common operations
- Review [SYSTEM_ARCHITECTURE.md](../SYSTEM_ARCHITECTURE.md) for system design
- See [README.md](../README.md) for project overview

---

**Happy Analyzing! 📊✨**
