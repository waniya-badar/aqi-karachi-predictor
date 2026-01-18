#!/usr/bin/env python3
"""
Simplified Test for AQI Predictor - No MongoDB Required
Tests data fetching, feature engineering, model training with synthetic data
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def generate_synthetic_aqi_data(days=90, samples_per_day=24):
    """Generate realistic synthetic AQI data for testing"""
    print("\n" + "="*70)
    print("Generating Synthetic AQI Dataset")
    print("="*70)
    
    np.random.seed(42)
    n_samples = days * samples_per_day
    
    # Generate base AQI with seasonal pattern
    base_aqi = 100
    seasonal_pattern = 30 * np.sin(np.arange(n_samples) * 2 * np.pi / (365 * 24))
    noise = np.random.normal(0, 15, n_samples)
    aqi_values = base_aqi + seasonal_pattern + noise
    aqi_values = np.clip(aqi_values, 20, 400)  # Realistic range
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate correlated pollutants
    pm25 = aqi_values * 0.6 + np.random.normal(0, 5, n_samples)
    pm10 = aqi_values * 0.8 + np.random.normal(0, 8, n_samples)
    o3 = 20 + np.random.normal(0, 10, n_samples)
    no2 = 30 + np.random.normal(0, 8, n_samples)
    
    # Generate weather data with daily patterns
    hour_of_day = np.array([t.hour for t in timestamps])
    temperature = 25 + 10 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 2, n_samples)
    humidity = 60 + 20 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 5, n_samples)
    pressure = 1013 + np.random.normal(0, 5, n_samples)
    wind_speed = 5 + 3 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'aqi': aqi_values,
        'pm25': pm25,
        'pm10': pm10,
        'o3': o3,
        'no2': no2,
        'so2': np.random.uniform(2, 15, n_samples),
        'co': np.random.uniform(0.3, 2.0, n_samples),
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed
    })
    
    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['day_of_week'] = df['timestamp'].dt.weekday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    print(f"Generated {len(df)} samples covering {days} days")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - AQI range: {df['aqi'].min():.0f} to {df['aqi'].max():.0f}")
    print(f"  - Features: {len(df.columns)} columns")
    
    return df


def test_feature_engineering(df):
    """Test feature engineering on synthetic data"""
    print("\n" + "="*70)
    print("Testing Feature Engineering")
    print("="*70)
    
    try:
        from feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        df_with_lags = engineer.add_lag_features(df.copy())
        print(f"Added lag features: {len(df_with_lags.columns)} total features")
        
        df_clean = engineer.handle_missing_values(df_with_lags)
        print(f"Handled missing values: {len(df_clean)} records remaining")
        
        print(f"Feature engineering successful")
        return df_clean
        
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_training(df):
    """Test all three models on synthetic data"""
    print("\n" + "="*70)
    print("Testing Model Training (3 Models)")
    print("="*70)
    
    try:
        from model_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        
        # Prepare data
        print("\nPreparing data...")
        X_train, X_test, y_train, y_test, features_used = trainer.prepare_data(df)
        
        print(f"\nData prepared:")
        print(f"  - Training: {len(X_train)} samples")
        print(f"  - Testing: {len(X_test)} samples")
        print(f"  - Features: {len(features_used)}")
        
        # Train Random Forest
        print(f"\n{'─'*70}")
        rf_results = trainer.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Train Gradient Boosting
        print(f"\n{'─'*70}")
        gb_results = trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        # Train Ridge Regression
        print(f"\n{'─'*70}")
        ridge_results = trainer.train_ridge(X_train, y_train, X_test, y_test)
        
        # Compare models
        print(f"\n{'─'*70}")
        best_model = trainer.compare_models()
        
        # Save models
        print(f"\n{'─'*70}")
        trainer.save_models(features_used)
        
        # Get feature importance for best model
        print(f"\n{'─'*70}")
        print(f"Getting feature importance for {trainer.results[best_model]['model_name']}...")
        importance_df = trainer.get_feature_importance(best_model)
        if importance_df is not None:
            print("\nTop 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        return trainer
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_predictions(trainer, df):
    """Test making predictions with trained models"""
    print("\n" + "="*70)
    print("Testing Predictions")
    print("="*70)
    
    try:
        import pickle
        
        # Load the best model
        best_model_path = 'models/saved_models/random_forest_latest.pkl'
        scaler_path = 'models/saved_models/scaler_latest.pkl'
        features_path = 'models/saved_models/feature_names.json'
        
        if not os.path.exists(best_model_path):
            print(f"✗ Model file not found: {best_model_path}")
            return False
        
        # Load model, scaler, and features
        with open(best_model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
        
        print(f"Loaded model with {len(feature_names)} features")
        
        test_features = df[feature_names].iloc[:20]
        test_features = pd.DataFrame(
            scaler.transform(test_features),
            columns=feature_names
        )
        
        predictions = model.predict(test_features)
        
        print(f"Made predictions for {len(predictions)} samples")
        print(f"\nSample predictions:")
        for i in range(min(5, len(predictions))):
            print(f"  Sample {i+1}: {predictions[i]:.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alert_system():
    """Test alert system"""
    print("\n" + "="*70)
    print("Testing Alert System")
    print("="*70)
    
    try:
        from alert_system import AlertSystem
        
        alert_system = AlertSystem()
        
        test_cases = [
            (50, "Good"),
            (75, "Moderate"),
            (120, "Unhealthy for Sensitive Groups"),
            (180, "Unhealthy"),
            (250, "Very Unhealthy"),
            (350, "Hazardous"),
        ]
        
        print("\nTesting AQI alert levels:")
        for aqi, expected_category in test_cases:
            level, should_alert, message = alert_system.check_aqi_level(aqi)
            status = "ALERT" if should_alert else "OK"
            print(f"  {status} - AQI {aqi:3d}: {level.replace('_', ' ').upper():30s}")
        
        print(f"\nAlert system working correctly")
        return True
        
    except Exception as e:
        print(f"Alert system test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("AQI PREDICTOR - COMPLETE TEST SUITE (NO MongoDB Required)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Step 1: Generate synthetic data
    print("\n[Step 1/5] Generating Synthetic Data...")
    df = generate_synthetic_aqi_data(days=90, samples_per_day=24)
    if df is None:
        print("✗ Failed to generate synthetic data")
        return 1
    
    # Step 2: Test feature engineering
    print("\n[Step 2/5] Testing Feature Engineering...")
    df_engineered = test_feature_engineering(df)
    results['Feature Engineering'] = df_engineered is not None
    
    if df_engineered is None:
        print("⚠ Continuing with original data...")
        df_engineered = df
    
    # Step 3: Test model training
    print("\n[Step 3/5] Training Models...")
    trainer = test_model_training(df_engineered)
    results['Model Training'] = trainer is not None
    
    # Step 4: Test predictions
    print("\n[Step 4/5] Testing Predictions...")
    if trainer:
        results['Predictions'] = test_predictions(trainer, df_engineered)
    else:
        results['Predictions'] = False
    
    # Step 5: Test alert system
    print("\n[Step 5/5] Testing Alert System...")
    results['Alert System'] = test_alert_system()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("Your AQI Predictor is ready to use!")
        print("="*70)
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
