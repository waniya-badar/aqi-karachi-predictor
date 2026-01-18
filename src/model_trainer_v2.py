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
        self.feature_names = None
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'aqi') -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        print(f"\nPreparing Data")
        print(f"Total records: {len(df)}")
        
        df = df[df[target_col].notna()].copy()
        print(f"Records with valid AQI: {len(df)}")
        
        feature_cols = [col for col in df.columns if col not in self.exclude_cols]
        
        valid_features = []
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                if df[col].notna().sum() / len(df) > 0.5:
                    valid_features.append(col)
        
        if not valid_features:
            print("No valid features found, using default features")
            valid_features = [col for col in feature_cols if col in df.columns]
        
        print(f"Selected {len(valid_features)} features: {valid_features[:5]}...")
        
        X = df[valid_features].copy()
        y = df[target_col].copy()
        
        X = X.fillna(X.median(numeric_only=True))
        
        if len(X) < 20:
            print(f"Warning: Only {len(X)} samples available")
        
        test_size = max(0.2, min(0.3, 10/len(X)))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=valid_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=valid_features, index=X_test.index)
        
        self.feature_names = valid_features
        
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
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        results = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        results['model_name'] = 'Random Forest'
        results['predictions'] = y_pred_test
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        print(f"Random Forest trained")
        print(f"  Train RMSE: {results['train_rmse']:.2f}")
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
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        results = self._evaluate_model(y_train, y_pred_train, y_test, y_pred_test)
        results['model_name'] = 'Gradient Boosting'
        results['predictions'] = y_pred_test
        
        self.models['gradient_boosting'] = model
        self.results['gradient_boosting'] = results
        
        print(f"Gradient Boosting trained")
        print(f"  Train RMSE: {results['train_rmse']:.2f}")
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
        results['predictions'] = y_pred_test
        
        self.models['ridge'] = model
        self.results['ridge'] = results
        
        print(f"Ridge Regression trained")
        print(f"  Train RMSE: {results['train_rmse']:.2f}")
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
        print(f"\nSaving Models")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f'{model_name}_{timestamp}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} to {model_path}")
            
            latest_path = os.path.join(self.models_dir, f'{model_name}_latest.pkl')
            with open(latest_path, 'wb') as f:
                pickle.dump(model, f)
        
        scaler_path = os.path.join(self.models_dir, 'scaler_latest.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler")
        
        features_path = os.path.join(self.models_dir, 'feature_names.json')
        with open(features_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"Saved feature names ({len(feature_names)} features)")
        
        # Save results to model registry
        registry_path = os.path.join('models', 'model_registry.json')
        
        # Load existing registry
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = []
        
        # Add new results
        results_copy = {}
        for model_name, result in self.results.items():
            result_copy = result.copy()
            if 'predictions' in result_copy:
                result_copy['predictions'] = result_copy['predictions'].tolist() if hasattr(result_copy['predictions'], 'tolist') else result_copy['predictions']
            results_copy[model_name] = result_copy
        
        registry.append({
            'timestamp': timestamp,
            'models': results_copy
        })
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"Updated model registry")
    
    def compare_models(self) -> str:
        """Compare all models and return best one"""
        print(f"\nModel Comparison Summary")
        
        best_model = None
        best_r2 = -float('inf')
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            print(f"\n{results['model_name']}:")
            print(f"  Train RMSE: {results['train_rmse']:.2f}")
            print(f"  Test RMSE: {results['test_rmse']:.2f}")
            print(f"  Test MAE: {results['test_mae']:.2f}")
            print(f"  Test R²: {results['test_r2']:.4f}")
            
            comparison_data.append({
                'model': results['model_name'],
                'train_rmse': results['train_rmse'],
                'test_rmse': results['test_rmse'],
                'test_mae': results['test_mae'],
                'test_r2': results['test_r2']
            })
            
            # Select best model by R² score
            if results['test_r2'] > best_r2:
                best_r2 = results['test_r2']
                best_model = model_name
        
        print(f"\nBest Model: {self.results[best_model]['model_name']}")
        print(f"  Test R² Score: {best_r2:.4f}\n")
        
        return best_model
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """Get feature importance for a model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


if __name__ == "__main__":
    print("Model trainer module loaded successfully")
