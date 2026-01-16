"""
Model Trainer - Trains ML models for AQI prediction
Implements 3 models: Random Forest, Gradient Boosting, and Ridge Regression
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelTrainer:
    """Trains and evaluates AQI prediction models"""
    
    def __init__(self, models_dir: str = 'models/saved_models'):
        """Initialize model trainer"""
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Define feature columns (excluding target and metadata)
        self.exclude_cols = ['aqi', 'timestamp', 'date', '_id', 'inserted_at', 
                             'station_name', 'latitude', 'longitude']
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'aqi') -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        print(f"\n=== Preparing Data ===")
        print(f"Total records: {len(df)}")
        
        # Remove records with missing target
        df = df[df[target_col].notna()].copy()
        print(f"Records with valid AQI: {len(df)}")
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in self.exclude_cols]
        
        # Also exclude any lag/rolling features that have too many NaN
        valid_features = []
        for col in feature_cols:
            if df[col].notna().sum() / len(df) > 0.7:  # Keep if >70% valid
                valid_features.append(col)
        
        print(f"Selected {len(valid_features)} features")
        
        # Prepare X and y
        X = df[valid_features].copy()
        y = df[target_col].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to keep feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=valid_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=valid_features)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, valid_features
    
    def train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest model"""
        print(f"\n=== Training Random Forest ===")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        results = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        results['model_name'] = 'Random Forest'
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        print(f"✓ Random Forest trained")
        print(f"  Test RMSE: {results['test_rmse']:.2f}")
        print(f"  Test R²: {results['test_r2']:.4f}")
        
        return results
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting model"""
        print(f"\n=== Training Gradient Boosting ===")
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        results = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        results['model_name'] = 'Gradient Boosting'
        
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = results
        
        print(f"✓ Gradient Boosting trained")
        print(f"  Test RMSE: {results['test_rmse']:.2f}")
        print(f"  Test R²: {results['test_r2']:.4f}")
        
        return results
    
    def train_ridge(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Ridge Regression model"""
        print(f"\n=== Training Ridge Regression ===")
        
        model = Ridge(alpha=1.0, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        results = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        results['model_name'] = 'Ridge Regression'
        
        self.models['ridge'] = model
        self.results['ridge'] = results
        
        print(f"✓ Ridge Regression trained")
        print(f"  Test RMSE: {results['test_rmse']:.2f}")
        print(f"  Test R²: {results['test_r2']:.4f}")
        
        return results
    
    def _evaluate_model(self, y_train, y_pred_train, y_test, y_pred_test) -> Dict:
        """Calculate evaluation metrics"""
        return {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_models(self, feature_names: List[str]):
        """Save all trained models and metadata"""
        print(f"\n=== Saving Models ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            # Save model
            model_path = os.path.join(self.models_dir, f'{model_name}_{timestamp}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved {model_name} to {model_path}")
            
            # Also save as "latest" for easy loading
            latest_path = os.path.join(self.models_dir, f'{model_name}_latest.pkl')
            with open(latest_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, 'scaler_latest.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler")
        
        # Save feature names
        features_path = os.path.join(self.models_dir, 'feature_names.json')
        with open(features_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"✓ Saved feature names")
        
        # Save results to model registry
        registry_path = os.path.join('models', 'model_registry.json')
        
        # Load existing registry
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = []
        
        # Add new results
        registry.append({
            'timestamp': timestamp,
            'models': self.results
        })
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✓ Updated model registry")
    
    def compare_models(self) -> str:
        """Compare all models and return best one"""
        print(f"\n=== Model Comparison ===")
        
        best_model = None
        best_rmse = float('inf')
        
        for model_name, results in self.results.items():
            print(f"\n{results['model_name']}:")
            print(f"  Test RMSE: {results['test_rmse']:.2f}")
            print(f"  Test MAE: {results['test_mae']:.2f}")
            print(f"  Test R²: {results['test_r2']:.4f}")
            
            if results['test_rmse'] < best_rmse:
                best_rmse = results['test_rmse']
                best_model = model_name
        
        print(f"\n✓ Best Model: {self.results[best_model]['model_name']}")
        print(f"  RMSE: {best_rmse:.2f}")
        
        return best_model


# Test
if __name__ == "__main__":
    print("Model trainer module loaded successfully")