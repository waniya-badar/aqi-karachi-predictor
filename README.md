# AQI Predictor for Karachi

Predict Air Quality Index for Karachi using a 100% Serverless Machine Learning Stack

---

## Project Overview

An end-to-end machine learning system for predicting Air Quality Index (AQI) in Karachi featuring:

- Real-time Data Collection - Hourly AQI data from Open-Meteo API (geo-based for Karachi: 24.8607, 67.0011)
- Cloud Storage - All data and models stored in MongoDB Atlas (serverless)
- Feature Engineering - Lag features, rolling statistics, temporal features (22 dynamic features)
- 3 ML Models - Random Forest, Gradient Boosting, Ridge Regression
- Training Dataset - 3 months (90 days) of historical Karachi data
- Automated Model Selection - Best model selected based on R² score
- Interactive Dashboard - Streamlit web application (loads models from MongoDB cloud)
- 3-Day Forecasting - Predictions for next 3 days using trained models
- Explainability - SHAP and LIME analysis
- CI/CD Automation - GitHub Actions for hourly/daily pipelines (continues from last stored state)

---

## Architecture

```
AQICN API --> GitHub Actions --> MongoDB Atlas
                                     |
              +----------------------+----------------------+
              |                      |                      |
        Hourly Pipeline        Daily Pipeline        Weekly Pipeline
        (Feature Data)         (Model Training)      (EDA/SHAP/LIME)
              |                      |                      |
              v                      v                      v
        MongoDB features      MongoDB models         Analysis Plots
              |                      |
              +----------+-----------+
                         |
                         v
                  Streamlit Dashboard
                  Inference Pipeline
```

---

## Project Structure

```
aqi-predictor-karachi/
├── src/                          # Core modules
│   ├── data_fetcher.py          # AQICN API integration
│   ├── feature_engineering.py   # Feature creation
│   ├── model_trainer.py         # ML model training
│   ├── model_explainer.py       # SHAP explainability
│   ├── alert_system.py          # AQI alert system
│   └── mongodb_handler.py       # MongoDB operations
├── pipelines/                    # Automated pipelines
│   ├── feature_pipeline.py      # Hourly data collection
│   ├── training_pipeline.py     # Daily model training
│   └── inference_pipeline.py    # Prediction generation
├── streamlit_app/               # Dashboard
│   └── app.py                   # Main Streamlit app
├── notebooks/                   # Analysis scripts
│   ├── eda_analysis.py         # Exploratory data analysis
│   └── explainability_analysis.py  # SHAP/LIME analysis
├── .github/workflows/          # GitHub Actions
│   ├── hourly-feature-pipeline.yml
│   ├── daily-training-pipeline.yml
│   ├── weekly-eda-explainability.yml
│   └── inference-pipeline.yml
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- MongoDB Atlas account
- AQICN API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/aqi-predictor-karachi.git
cd aqi-predictor-karachi
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
# Create .env file with:
MONGODB_URI=your_mongodb_connection_string
AQICN_API_KEY=your_aqicn_api_key
```

### Run Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

Access at: http://localhost:8501

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data Source | AQICN API |
| Database | MongoDB Atlas (Cloud) |
| ML Models | Scikit-learn |
| Feature Engineering | Pandas, NumPy |
| Explainability | SHAP, LIME |
| Dashboard | Streamlit, Plotly |
| CI/CD | GitHub Actions |
| Hosting | Serverless (Cloud) |

---

## Machine Learning Models

### Models Implemented

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| Random Forest | 0.8923 | 4.00 | 5.52 |
| Gradient Boosting | 0.8866 | 4.18 | 5.67 |
| Ridge Regression | 0.8246 | 5.51 | 7.05 |

Best model is automatically selected based on highest R² score.

### Evaluation Metrics

- R² (R-squared): Proportion of variance explained (higher is better)
- MAE (Mean Absolute Error): Average prediction error in AQI points
- RMSE (Root Mean Squared Error): Penalizes larger errors more heavily

---

## Features Engineered

### Weather Features
- Temperature
- Humidity
- Pressure
- Wind Speed
- Visibility

### Lag Features (Historical AQI)
- 1-hour lag
- 3-hour lag
- 6-hour lag
- 12-hour lag
- 24-hour lag

### Rolling Statistics
- 6-hour rolling mean
- 12-hour rolling mean
- 24-hour rolling mean
- 24-hour rolling standard deviation

### Change Features
- 1-hour AQI change
- 6-hour AQI change

Total: 16 features used for prediction

---

## Data Pipeline

### Hourly Feature Pipeline
1. Fetch current AQI data from AQICN API
2. Create features (weather + AQI)
3. Store in MongoDB features collection
4. Runs every hour via GitHub Actions

### Daily Training Pipeline
1. Fetch 90 days (3 months) of historical data from MongoDB cloud
2. Create lag and rolling features (22 total features)
3. Train 3 models (Random Forest, Gradient Boosting, Ridge Regression)
4. Evaluate using R², MAE, RMSE
5. Select best model automatically (currently Gradient Boosting with R²=0.9919)
6. Save all models + metrics + scaler to MongoDB cloud (no local files)
7. Log training run to training_history collection
8. Runs daily at 2 AM UTC via GitHub Actions

### Inference Pipeline
1. Load best model from MongoDB cloud (serverless)
2. Fetch current conditions from Open-Meteo API
3. Generate 3-day forecast (next 72 hours)
4. Save predictions to MongoDB cloud
5. Runs every 6 hours via GitHub Actions

### Weekly EDA Pipeline
1. Load all data from MongoDB
2. Generate EDA plots
3. Run SHAP/LIME analysis
4. Runs every Sunday

---

## AQI Categories

| Category | AQI Range | Color | Health Impact |
|----------|-----------|-------|---------------|
| Good | 0-50 | Green | No health impacts |
| Moderate | 51-100 | Yellow | Sensitive groups may be affected |
| Unhealthy for Sensitive Groups | 101-150 | Orange | Health impacts for sensitive groups |
| Unhealthy | 151-200 | Red | Health impacts for all |
| Very Unhealthy | 201-300 | Purple | Serious health effects |
| Hazardous | 301+ | Maroon | Emergency conditions |

---

## GitHub Actions Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| Hourly Feature Pipeline | Every hour | Collect AQI data |
| Daily Training Pipeline | Daily 2 AM UTC | Train models |
| Weekly EDA | Sundays | Analysis and plots |
| Inference Pipeline | Every 6 hours | Generate predictions |

---

## Environment Variables

```bash
# Required
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/aqi_karachi
AQICN_API_KEY=your_api_key

# Optional
ALERT_EMAIL=your_email@example.com
```

---

## API Usage

### Data Fetcher
```python
from src.data_fetcher import AQICNFetcher

fetcher = AQICNFetcher()
data = fetcher.fetch_current_data()
# Returns: {'aqi': 73, 'pm25': 25, 'temperature': 22, ...}
```

### MongoDB Handler
```python
from src.mongodb_handler import MongoDBHandler

db = MongoDBHandler()
df = db.get_training_data(days=365)
model = db.get_best_model()
```

### Model Trainer
```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train_all_models(df, db_handler)
# Trains RF, GB, Ridge and saves best to MongoDB
```

---

## Dashboard Features

Current AQI and pollutant levels from the latest feature store data
3-day forecast using the latest feature store data
Historical AQI graph (always shows last 90 days)
Model selection (choose between 3 models)
Model metrics display (R², MAE, RMSE)
Pollutant levels breakdown

---

## Data Storage

All data is stored in MongoDB Atlas (cloud):

| Collection | Content |
|------------|---------|
| features | Hourly AQI and weather data |
| models | Trained model binaries and metadata |
| training_history | Training run logs |
| predictions | Generated forecasts |

No local data storage required.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

---

## License

MIT License - see LICENSE file for details.

---

## References

- AQICN API: https://aqicn.org/api/
- Scikit-learn: https://scikit-learn.org/
- Streamlit: https://docs.streamlit.io/
- SHAP: https://shap.readthedocs.io/

---

Last Updated: January 24, 2026

---

## Workflow Automation

GitHub Actions workflows automatically:
- Continue from last stored state (no manual intervention needed)
- Resume hourly feature collection from where it left off
- Train models on growing dataset (90-day rolling window)
- Generate predictions using latest trained models
- All workflows work automatically after pushing to GitHub (no need to disable/enable)
