"""
Feature Engineering - Creates ML features from raw data
Transforms raw AQI data into features for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional


class FeatureEngineer:
    """Creates features from raw AQI data"""
    
    def __init__(self):
        """Initialize feature engineer"""
        pass
    
    def create_features(self, raw_data: Dict) -> Optional[Dict]:
        """
        Create features from raw AQI data
        
        Args:
            raw_data: Dictionary with raw data from API
        
        Returns:
            Dictionary with engineered features
        """
        try:
            if not raw_data or 'timestamp' not in raw_data:
                print("Invalid raw data provided")
                return None
            
            timestamp = raw_data['timestamp']
            
            # Initialize features dictionary
            # Initialize features dictionary
            features = {
                'timestamp': timestamp,
                'date': timestamp.strftime('%Y-%m-%d'),
            }
                        
            features['hour'] = timestamp.hour
            features['day'] = timestamp.day
            features['month'] = timestamp.month
            features['year'] = timestamp.year
            features['day_of_week'] = timestamp.weekday()
            features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
            
            if 6 <= timestamp.hour < 12:
                features['time_of_day'] = 1
            elif 12 <= timestamp.hour < 18:
                features['time_of_day'] = 2
            elif 18 <= timestamp.hour < 22:
                features['time_of_day'] = 3
            else:
                features['time_of_day'] = 4
            
            features['aqi'] = raw_data.get('aqi', None)
            
            features['pm25'] = raw_data.get('pm25', None)
            features['pm10'] = raw_data.get('pm10', None)
            features['o3'] = raw_data.get('o3', None)
            features['no2'] = raw_data.get('no2', None)
            
            features['so2'] = raw_data.get('so2', None)
            features['co'] = raw_data.get('co', None)
            
            features['temperature'] = raw_data.get('temperature', None)
            features['humidity'] = raw_data.get('humidity', None)
            features['pressure'] = raw_data.get('pressure', None)
            features['wind_speed'] = raw_data.get('wind_speed', None)
            
            features['latitude'] = raw_data.get('latitude', 24.8607)
            features['longitude'] = raw_data.get('longitude', 67.0011)
            
            print(f"Created {len(features)} features for {timestamp}")
            return features
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return None
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag features (previous values) to DataFrame
        This helps model understand trends over time
        
        Args:
            df: DataFrame with features sorted by timestamp
        
        Returns:
            DataFrame with lag features added
        """
        try:
            # Sort by timestamp to ensure correct order
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Create lag features for key pollutants and AQI
            lag_periods = [1, 3, 6, 12, 24]  # Hours ago
            lag_columns = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity']
            
            for col in lag_columns:
                if col in df.columns:
                    for lag in lag_periods:
                        df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
            
            # Calculate rolling statistics (trends)
            window_sizes = [6, 12, 24]  # Hours
            
            for window in window_sizes:
                if 'aqi' in df.columns:
                    df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).mean()
                    df[f'aqi_rolling_std_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).std()
                
                if 'pm25' in df.columns:
                    df[f'pm25_rolling_mean_{window}h'] = df['pm25'].rolling(window=window, min_periods=1).mean()
            
            # Calculate change rates
            if 'aqi' in df.columns:
                df['aqi_change_1h'] = df['aqi'].diff(1)
                df['aqi_change_6h'] = df['aqi'].diff(6)
            
            print(f"Added lag features to {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Error adding lag features: {e}")
            return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with missing values handled
        """
        try:
            # For pollutants and weather, use forward fill then backward fill
            pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
            weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
            
            for col in pollutant_cols + weather_cols:
                if col in df.columns:
                    # Forward fill (use previous value)
                    df[col] = df[col].fillna(method='ffill')
                    # Backward fill for remaining
                    df[col] = df[col].fillna(method='bfill')
                    # If still NaN, use median
                    df[col] = df[col].fillna(df[col].median())
            
            # For lag features, drop rows with too many missing values
            # (initial rows won't have lag values)
            threshold = len(df.columns) * 0.3  # Drop if >30% values missing
            df = df.dropna(thresh=threshold)
            
            # Fill remaining NaN with 0
            df = df.fillna(0)
            
            print(f" Handled missing values, {len(df)} records remaining")
            return df
            
        except Exception as e:
            print(f" Error handling missing values: {e}")
            return df
    
    def get_feature_names(self) -> list:
        """Get list of all feature names"""
        base_features = [
            'hour', 'day', 'month', 'year', 'day_of_week', 'is_weekend', 'time_of_day',
            'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
            'temperature', 'humidity', 'pressure', 'wind_speed',
            'latitude', 'longitude'
        ]
        return base_features


# Test feature engineering
if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Test with sample data
    sample_data = {
        'timestamp': datetime.now(),
        'aqi': 150,
        'pm25': 55,
        'pm10': 90,
        'o3': 40,
        'no2': 30,
        'temperature': 28,
        'humidity': 65,
        'pressure': 1010,
        'wind_speed': 12
    }
    
    features = engineer.create_features(sample_data)
    
    print("\n=== Sample Features ===")
    for key, value in features.items():
        print(f"{key}: {value}")