#!/usr/bin/env python3
"""
Inference Pipeline - Karachi AQI Prediction
Loads models from MongoDB and generates predictions for current/future AQI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.mongodb_handler import MongoDBHandler
from src.data_fetcher import AQICNFetcher
from src.feature_engineering import FeatureEngineer


def load_best_model():
    """Load the best performing model from MongoDB"""
    print("\n📥 Loading best model from MongoDB...")
    
    db_handler = MongoDBHandler()
    model_doc = db_handler.get_best_model()
    
    if model_doc is None:
        print("  [FAIL] Could not load model from MongoDB")
        db_handler.close()
        return None, None, None, None, None
    
    # Deserialize model and scaler
    model = pickle.loads(model_doc['model_binary'])
    scaler = pickle.loads(model_doc['scaler_binary'])
    feature_names = model_doc.get('feature_names', [])
    metrics = model_doc.get('metrics', {})
    model_name = model_doc.get('model_name', 'unknown')
    
    print(f"  [OK] Loaded {model_name}")
    print(f"       R²: {metrics.get('test_r2', 0):.4f}, MAE: {metrics.get('test_mae', 0):.2f}, RMSE: {metrics.get('test_rmse', 0):.2f}")
    
    db_handler.close()
    return model, scaler, feature_names, metrics, model_name


def get_current_conditions():
    """Fetch current AQI and weather conditions from latest feature in MongoDB"""
    print("\n🌍 Loading current conditions from MongoDB...")
    
    db_handler = MongoDBHandler()
    latest_df = db_handler.get_latest_features(limit=1)
    db_handler.close()
    
    if latest_df is None or len(latest_df) == 0:
        print("  [FAIL] No features found in MongoDB")
        return None
    
    latest_features = latest_df.iloc[0].to_dict()
    
    print(f"  [OK] Current AQI: {latest_features.get('aqi', 'N/A')}")
    print(f"       Temperature: {latest_features.get('temperature', 'N/A')}°C")
    print(f"       Humidity: {latest_features.get('humidity', 'N/A')}%")
    print(f"       Timestamp: {latest_features.get('timestamp', 'N/A')}")
    
    return latest_features


def get_recent_aqi_history():
    """Get recent AQI values for lag features"""
    print("\n📊 Fetching recent AQI history...")
    
    db_handler = MongoDBHandler()
    df = db_handler.get_training_data(days=2)  # Last 2 days for lag features
    db_handler.close()
    
    if df is None or len(df) < 24:
        print("  [WARN] Insufficient history, using defaults")
        return None
    
    df = df.sort_values('timestamp', ascending=False)
    print(f"  [OK] Loaded {len(df)} recent records")
    
    return df


def predict_future_aqi(model, scaler, current_features, feature_names, history_df=None, days=3):
    """Generate AQI predictions for upcoming days"""
    predictions = []
    
    # Get recent AQI values for lag features
    if history_df is not None and len(history_df) >= 24:
        aqi_values = history_df['aqi'].values
        recent_aqi = aqi_values[0] if len(aqi_values) > 0 else current_features.get('aqi', 70)
    else:
        recent_aqi = current_features.get('aqi', 70)
        aqi_values = [recent_aqi] * 25  # Dummy history
    
    # Initialize lag values
    aqi_lag_1h = aqi_values[0] if len(aqi_values) > 0 else recent_aqi
    aqi_lag_3h = aqi_values[2] if len(aqi_values) > 2 else recent_aqi
    aqi_lag_6h = aqi_values[5] if len(aqi_values) > 5 else recent_aqi
    aqi_lag_12h = aqi_values[11] if len(aqi_values) > 11 else recent_aqi
    aqi_lag_24h = aqi_values[23] if len(aqi_values) > 23 else recent_aqi
    
    # Calculate rolling means
    aqi_rolling_6h = np.mean(aqi_values[:6]) if len(aqi_values) >= 6 else recent_aqi
    aqi_rolling_24h = np.mean(aqi_values[:24]) if len(aqi_values) >= 24 else recent_aqi
    
    for day in range(1, days + 1):
        future_date = datetime.now() + timedelta(days=day)
        
        # Build feature vector
        feature_vector = {}
        
        for feature in feature_names:
            if feature == 'aqi_lag_1h':
                feature_vector[feature] = aqi_lag_1h
            elif feature == 'aqi_lag_3h':
                feature_vector[feature] = aqi_lag_3h
            elif feature == 'aqi_lag_6h':
                feature_vector[feature] = aqi_lag_6h
            elif feature == 'aqi_lag_12h':
                feature_vector[feature] = aqi_lag_12h
            elif feature == 'aqi_lag_24h':
                feature_vector[feature] = aqi_lag_24h
            elif feature == 'aqi_rolling_mean_6h':
                feature_vector[feature] = aqi_rolling_6h
            elif feature == 'aqi_rolling_mean_24h':
                feature_vector[feature] = aqi_rolling_24h
            elif feature == 'aqi_change_1h':
                feature_vector[feature] = aqi_lag_1h - aqi_lag_3h
            elif feature == 'aqi_change_24h':
                feature_vector[feature] = aqi_lag_1h - aqi_lag_24h
            elif feature == 'hour':
                feature_vector[feature] = 12  # Midday prediction
            elif feature == 'day_of_week':
                feature_vector[feature] = future_date.weekday()
            elif feature == 'is_weekend':
                feature_vector[feature] = 1 if future_date.weekday() >= 5 else 0
            elif feature == 'month':
                feature_vector[feature] = future_date.month
            elif feature in current_features:
                # Add small variation for weather features
                base_value = current_features[feature]
                if isinstance(base_value, (int, float)):
                    variation = np.random.uniform(-0.05, 0.05) * base_value * day
                    feature_vector[feature] = base_value + variation
                else:
                    feature_vector[feature] = base_value
            else:
                feature_vector[feature] = 0
        
        # Create DataFrame and predict
        X = pd.DataFrame([feature_vector])[feature_names]
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        predicted_aqi = model.predict(X_scaled)[0]
        predicted_aqi = max(0, min(500, predicted_aqi))  # Clamp to valid range
        
        predictions.append({
            'day': day,
            'date': future_date.strftime('%Y-%m-%d'),
            'weekday': future_date.strftime('%A'),
            'aqi': round(predicted_aqi, 1),
            'category': get_aqi_category(predicted_aqi)
        })
        
        # Update lag values for next iteration
        aqi_lag_24h = aqi_lag_12h
        aqi_lag_12h = aqi_lag_6h
        aqi_lag_6h = aqi_lag_3h
        aqi_lag_3h = aqi_lag_1h
        aqi_lag_1h = predicted_aqi
        
        # Update rolling means
        aqi_rolling_6h = (aqi_rolling_6h * 5 + predicted_aqi) / 6
        aqi_rolling_24h = (aqi_rolling_24h * 23 + predicted_aqi) / 24
    
    return predictions


def get_aqi_category(aqi):
    """Get AQI category based on value"""
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Moderate 🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 🟠"
    elif aqi <= 200:
        return "Unhealthy 🔴"
    elif aqi <= 300:
        return "Very Unhealthy 🟣"
    else:
        return "Hazardous ⚫"


def save_predictions_to_mongodb(predictions, current_aqi):
    """Save predictions to MongoDB for tracking (Cloud Storage)"""
    db_handler = MongoDBHandler()

    prediction_record = {
        'timestamp': datetime.now(),
        'current_aqi': current_aqi,
        'predictions': predictions,
        'model_used': 'best_model',
        'prediction_horizon_days': len(predictions)
    }

    try:
        success = db_handler.save_prediction(prediction_record)
        if success:
            print("  [OK] Predictions saved to MongoDB (predictions collection)")
        else:
            print("  [WARN] Could not save predictions")
        print(f"  [WARN] Could not save predictions: {e}")
    
    db_handler.close()


def main():
    """Main inference pipeline"""
    print("\n" + "="*70)
    print("🔮 AQI PREDICTION INFERENCE PIPELINE")
    print("    Karachi Air Quality Forecast")
    print("="*70)
    
    # Load best model
    model, scaler, feature_names, metrics, model_name = load_best_model()
    if model is None:
        print("\n❌ Failed to load model. Exiting.")
        return False
    
    # Get current conditions
    current_features = get_current_conditions()
    if current_features is None:
        print("\n❌ Failed to fetch current conditions. Exiting.")
        return False
    
    # Get recent history for lag features
    history_df = get_recent_aqi_history()
    
    # Generate predictions
    print("\n🔮 Generating 3-day forecast...")
    predictions = predict_future_aqi(
        model, scaler, current_features, feature_names, 
        history_df=history_df, days=3
    )
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    
    current_aqi = current_features.get('aqi', 'N/A')
    print(f"\n  Current AQI: {current_aqi} ({get_aqi_category(current_aqi) if isinstance(current_aqi, (int, float)) else 'N/A'})")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n  3-Day Forecast:")
    print("  " + "-"*50)
    
    for pred in predictions:
        print(f"  Day {pred['day']} | {pred['date']} ({pred['weekday'][:3]})")
        print(f"        AQI: {pred['aqi']} - {pred['category']}")
        print()
    
    # Trend analysis
    aqi_values = [current_aqi] + [p['aqi'] for p in predictions]
    if all(isinstance(v, (int, float)) for v in aqi_values):
        if aqi_values[-1] > aqi_values[0] * 1.1:
            trend = "📈 WORSENING - AQI expected to increase"
        elif aqi_values[-1] < aqi_values[0] * 0.9:
            trend = "📉 IMPROVING - AQI expected to decrease"
        else:
            trend = "➡️ STABLE - AQI expected to remain similar"
        print(f"  Trend: {trend}")
    
    # Save to MongoDB
    print("\n💾 Saving predictions...")
    save_predictions_to_mongodb(predictions, current_aqi)
    
    print("\n" + "="*70)
    print("✅ INFERENCE PIPELINE COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
