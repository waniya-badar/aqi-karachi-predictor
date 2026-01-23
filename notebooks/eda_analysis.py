"""
Exploratory Data Analysis - Karachi AQI
Analyzes all data from MongoDB cloud database
Displays results for all 3 trained models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import pickle
warnings.filterwarnings('ignore')

from src.mongodb_handler import MongoDBHandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create plots directory
os.makedirs('notebooks/plots', exist_ok=True)

print("="*70)
print("EXPLORATORY DATA ANALYSIS - KARACHI AQI")
print("="*70)

print("\n1. LOADING ALL DATA FROM MONGODB")

db_handler = MongoDBHandler()

# Get ALL data from MongoDB (use large number of days to get everything)
df = db_handler.get_training_data(days=365)  # Get up to 1 year of data

if df is None or len(df) == 0:
    print("No data available. Run feature pipeline first.")
    sys.exit(1)

# Calculate date range
date_range_days = (df['timestamp'].max() - df['timestamp'].min()).days
print(f"Loaded {len(df)} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Data span: {date_range_days} days (~{date_range_days/30:.1f} months)")
print(f"Features: {len(df.columns)} columns")


print("\n2. BASIC STATISTICS")

print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes.value_counts())

print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print(missing)
else:
    print("No missing values!")

print("\nAQI Statistics:")
print(df['aqi'].describe())

print("\n3. TEMPORAL ANALYSIS")

df['date'] = pd.to_datetime(df['timestamp']).dt.date
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_name'] = pd.to_datetime(df['timestamp']).dt.day_name()
df['month_name'] = pd.to_datetime(df['timestamp']).dt.month_name()

# AQI Time Series
plt.figure(figsize=(14, 6))
plt.plot(df['timestamp'], df['aqi'], linewidth=1, alpha=0.7)
plt.axhline(y=50, color='green', linestyle='--', label='Good')
plt.axhline(y=100, color='yellow', linestyle='--', label='Moderate')
plt.axhline(y=150, color='orange', linestyle='--', label='Unhealthy for Sensitive')
plt.axhline(y=200, color='red', linestyle='--', label='Unhealthy')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title(f'AQI Time Series - Karachi ({date_range_days} days of data)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/plots/aqi_timeseries.png', dpi=300, bbox_inches='tight')
print("Saved: aqi_timeseries.png")

# Hourly Patterns
plt.figure(figsize=(12, 6))
hourly_aqi = df.groupby('hour')['aqi'].agg(['mean', 'std'])
plt.errorbar(hourly_aqi.index, hourly_aqi['mean'], yerr=hourly_aqi['std'], 
             marker='o', capsize=5, capthick=2)
plt.xlabel('Hour of Day')
plt.ylabel('Average AQI')
plt.title('Average AQI by Hour of Day', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('notebooks/plots/hourly_pattern.png', dpi=300, bbox_inches='tight')
print("Saved: hourly_pattern.png")

# Day of Week Patterns
plt.figure(figsize=(12, 6))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_aqi = df.groupby('day_name')['aqi'].mean().reindex(day_order)
plt.bar(daily_aqi.index, daily_aqi.values, color='steelblue', alpha=0.7)
plt.xlabel('Day of Week')
plt.ylabel('Average AQI')
plt.title('Average AQI by Day of Week', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('notebooks/plots/weekly_pattern.png', dpi=300, bbox_inches='tight')
print("Saved: weekly_pattern.png")

print("\n4. POLLUTANT ANALYSIS")

pollutants = ['pm25', 'pm10', 'o3', 'no2']
available_pollutants = [p for p in pollutants if p in df.columns]

if len(available_pollutants) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, pollutant in enumerate(available_pollutants[:4]):
        axes[idx].scatter(df[pollutant], df['aqi'], alpha=0.5, s=20)
        
        corr = df[[pollutant, 'aqi']].corr().iloc[0, 1]
        
        z = np.polyfit(df[pollutant].dropna(), df.loc[df[pollutant].notna(), 'aqi'], 1)
        p = np.poly1d(z)
        axes[idx].plot(df[pollutant], p(df[pollutant]), "r--", alpha=0.8, linewidth=2)
        
        axes[idx].set_xlabel(f'{pollutant.upper()} (µg/m³)')
        axes[idx].set_ylabel('AQI')
        axes[idx].set_title(f'AQI vs {pollutant.upper()} (Corr: {corr:.3f})', fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/plots/pollutant_correlations.png', dpi=300, bbox_inches='tight')
    print("Saved: pollutant_correlations.png")

    plt.figure(figsize=(12, 6))
    pollutant_data = df[available_pollutants].dropna()
    plt.boxplot([pollutant_data[p] for p in available_pollutants], 
                labels=[p.upper() for p in available_pollutants])
    plt.ylabel('Concentration (µg/m³)')
    plt.title('Distribution of Pollutants', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('notebooks/plots/pollutants_boxplot.png', dpi=300, bbox_inches='tight')
    print("Saved: pollutants_boxplot.png")

print("\n5. WEATHER CORRELATION ANALYSIS")

weather_vars = ['temperature', 'humidity', 'pressure', 'wind_speed']
available_weather = [w for w in weather_vars if w in df.columns and df[w].notna().sum() > 10]

if len(available_weather) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, weather in enumerate(available_weather[:4]):
        axes[idx].scatter(df[weather], df['aqi'], alpha=0.5, s=20, c='coral')
        corr = df[[weather, 'aqi']].corr().iloc[0, 1]
        axes[idx].set_xlabel(f'{weather.replace("_", " ").title()}')
        axes[idx].set_ylabel('AQI')
        axes[idx].set_title(f'AQI vs {weather.replace("_", " ").title()} (Corr: {corr:.3f})', 
                           fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('notebooks/plots/weather_correlations.png', dpi=300, bbox_inches='tight')
    print("Saved: weather_correlations.png")

print("\n6. CORRELATION MATRIX")

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['timestamp', 'year']]

if len(numerical_cols) > 5:
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('notebooks/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_matrix.png")
    
    aqi_corr = corr_matrix['aqi'].sort_values(ascending=False)
    print("\nTop 10 Features Correlated with AQI:")
    print(aqi_corr.head(10))

print("\n7. AQI DISTRIBUTION ANALYSIS")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(df['aqi'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(df['aqi'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["aqi"].mean():.1f}')
axes[0].axvline(df['aqi'].median(), color='green', linestyle='--', 
                label=f'Median: {df["aqi"].median():.1f}')
axes[0].set_xlabel('AQI')
axes[0].set_ylabel('Frequency')
axes[0].set_title('AQI Distribution', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].boxplot(df['aqi'], vert=True)
axes[1].set_ylabel('AQI')
axes[1].set_title('AQI Box Plot', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/plots/aqi_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: aqi_distribution.png")

aqi_categories = pd.cut(df['aqi'], 
                        bins=[0, 50, 100, 150, 200, 300, 500],
                        labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 
                               'Unhealthy', 'Very Unhealthy', 'Hazardous'])

plt.figure(figsize=(10, 6))
category_counts = aqi_categories.value_counts()
colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']
plt.bar(category_counts.index, category_counts.values, color=colors[:len(category_counts)], alpha=0.7)
plt.xlabel('AQI Category')
plt.ylabel('Count')
plt.title('Distribution of AQI Categories', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('notebooks/plots/aqi_categories.png', dpi=300, bbox_inches='tight')
print("Saved: aqi_categories.png")

print("\n8. MODEL PERFORMANCE ANALYSIS")

print("\nLoading trained models from MongoDB...")
db_handler_models = MongoDBHandler()
models = {}
model_metrics = {}

model_names = ['random_forest', 'gradient_boosting', 'ridge']

for model_name in model_names:
    model_data = db_handler_models.get_model(model_name)
    if model_data:
        models[model_name] = pickle.loads(model_data['model_binary'])
        scaler = pickle.loads(model_data['scaler_binary'])
        metrics = model_data.get('metrics', {})
        model_metrics[model_name] = {
            'scaler': scaler,
            'r2': metrics.get('test_r2', 0),
            'mae': metrics.get('test_mae', 0),
            'rmse': metrics.get('test_rmse', 0),
            'train_r2': metrics.get('train_r2', 0)
        }
        print(f"  [OK] Loaded {model_name}")
        print(f"       R²: {model_metrics[model_name]['r2']:.4f}, MAE: {model_metrics[model_name]['mae']:.2f}, RMSE: {model_metrics[model_name]['rmse']:.2f}")
    else:
        print(f"  [FAIL] Could not load {model_name}")

db_handler_models.close()

if len(models) > 0:
    # Prepare feature columns for prediction
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'date', 'hour', 'day_name', 'month_name', 'aqi']]
    
    # Get subset with all required features
    df_features = df[feature_cols + ['aqi']].dropna()
    
    if len(df_features) > 100:
        # Split for evaluation
        X = df_features[feature_cols]
        y = df_features['aqi']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"\nEvaluating models on {len(X_test)} test samples...")
        
        # Evaluate each model
        predictions = {}
        for model_name, model in models.items():
            scaler = model_metrics[model_name]['scaler']
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            predictions[model_name] = y_pred
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
        
        # Plot model comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (model_name, y_pred) in enumerate(predictions.items()):
            axes[idx].scatter(y_test, y_pred, alpha=0.5, s=20)
            axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                          'r--', lw=2, label='Perfect Prediction')
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            axes[idx].set_xlabel('Actual AQI')
            axes[idx].set_ylabel('Predicted AQI')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nR²={r2:.4f}, MAE={mae:.2f}', 
                               fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('notebooks/plots/model_predictions_comparison.png', dpi=300, bbox_inches='tight')
        print("\nSaved: model_predictions_comparison.png")
        
        # Create metrics comparison bar chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        model_names_list = list(predictions.keys())
        r2_scores = [r2_score(y_test, predictions[m]) for m in model_names_list]
        mae_scores = [mean_absolute_error(y_test, predictions[m]) for m in model_names_list]
        rmse_scores = [np.sqrt(mean_squared_error(y_test, predictions[m])) for m in model_names_list]
        
        colors = ['steelblue', 'coral', 'mediumseagreen']
        
        axes[0].bar(range(len(model_names_list)), r2_scores, color=colors, alpha=0.7)
        axes[0].set_xticks(range(len(model_names_list)))
        axes[0].set_xticklabels([m.replace('_', ' ').title() for m in model_names_list], rotation=45, ha='right')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² Score Comparison', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([min(r2_scores) - 0.01, 1.0])
        
        axes[1].bar(range(len(model_names_list)), mae_scores, color=colors, alpha=0.7)
        axes[1].set_xticks(range(len(model_names_list)))
        axes[1].set_xticklabels([m.replace('_', ' ').title() for m in model_names_list], rotation=45, ha='right')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('MAE Comparison (lower is better)', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        axes[2].bar(range(len(model_names_list)), rmse_scores, color=colors, alpha=0.7)
        axes[2].set_xticks(range(len(model_names_list)))
        axes[2].set_xticklabels([m.replace('_', ' ').title() for m in model_names_list], rotation=45, ha='right')
        axes[2].set_ylabel('RMSE')
        axes[2].set_title('RMSE Comparison (lower is better)', fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('notebooks/plots/model_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: model_metrics_comparison.png")
        
        # Find best model
        best_model_name = model_names_list[np.argmax(r2_scores)]
        print(f"\nBest Model: {best_model_name.replace('_', ' ').title()} (R²={max(r2_scores):.4f})")
    else:
        print("\nInsufficient data with features for model evaluation")
else:
    print("\nNo models loaded - skipping model evaluation")

print("\n9. KEY INSIGHTS")

print(f"\nData Summary:")
print(f"  Total Records: {len(df)}")
print(f"  Date Range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
print(f"  Data Span: {date_range_days} days (~{date_range_days/30:.1f} months)")

print(f"\nAQI Summary:")
print(f"  Average AQI: {df['aqi'].mean():.1f}")
print(f"  Median AQI: {df['aqi'].median():.1f}")
print(f"  Min AQI: {df['aqi'].min():.1f}")
print(f"  Max AQI: {df['aqi'].max():.1f}")
print(f"  Std Dev: {df['aqi'].std():.1f}")

print(f"\nTemporal Patterns:")
worst_hour = df.groupby('hour')['aqi'].mean().idxmax()
best_hour = df.groupby('hour')['aqi'].mean().idxmin()
print(f"  Worst hour: {worst_hour}:00 (AQI: {df.groupby('hour')['aqi'].mean()[worst_hour]:.1f})")
print(f"  Best hour: {best_hour}:00 (AQI: {df.groupby('hour')['aqi'].mean()[best_hour]:.1f})")

if 'day_name' in df.columns:
    worst_day = df.groupby('day_name')['aqi'].mean().idxmax()
    best_day = df.groupby('day_name')['aqi'].mean().idxmin()
    print(f"  Worst day: {worst_day}")
    print(f"  Best day: {best_day}")

print(f"\nHealth Concerns:")
unhealthy_hours = (df['aqi'] > 150).sum()
hazardous_hours = (df['aqi'] > 300).sum()
print(f"  Hours with unhealthy AQI (>150): {unhealthy_hours} ({unhealthy_hours/len(df)*100:.1f}%)")
print(f"  Hours with hazardous AQI (>300): {hazardous_hours} ({hazardous_hours/len(df)*100:.1f}%)")

if len(available_pollutants) > 0:
    print(f"\nTop Pollutant Correlations with AQI:")
    for pollutant in available_pollutants:
        corr = df[[pollutant, 'aqi']].corr().iloc[0, 1]
        print(f"  {pollutant.upper()}: {corr:.3f}")

print("\n10. EXPORTING SUMMARY")

summary = {
    'Total Records': len(df),
    'Date Range Start': str(df['timestamp'].min()),
    'Date Range End': str(df['timestamp'].max()),
    'Data Span Days': date_range_days,
    'Average AQI': df['aqi'].mean(),
    'Median AQI': df['aqi'].median(),
    'Max AQI': df['aqi'].max(),
    'Min AQI': df['aqi'].min(),
    'Std Dev': df['aqi'].std(),
    'Unhealthy Hours (%)': (df['aqi'] > 150).sum() / len(df) * 100,
    'Hazardous Hours (%)': (df['aqi'] > 300).sum() / len(df) * 100
}

# Add model metrics to summary if available
if len(models) > 0 and len(df_features) > 100:
    for model_name in predictions.keys():
        y_pred = predictions[model_name]
        summary[f'{model_name}_r2'] = r2_score(y_test, y_pred)
        summary[f'{model_name}_mae'] = mean_absolute_error(y_test, y_pred)
        summary[f'{model_name}_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))

summary_df = pd.DataFrame([summary])
summary_df.to_csv('notebooks/eda_summary.csv', index=False)
print("Saved: eda_summary.csv")

db_handler.close()

print("\n" + "="*70)
print("EDA COMPLETE!")
print("="*70)
print(f"\nAll plots saved to: notebooks/plots/")
print(f"Summary saved to: notebooks/eda_summary.csv")
print(f"\nAnalyzed {len(df)} records spanning {date_range_days} days")
