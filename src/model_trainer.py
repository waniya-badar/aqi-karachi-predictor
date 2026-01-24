"""
Model Trainer - Trains ML models for AQI prediction
Implements 3 models: Random Forest, Gradient Boosting, and Ridge Regression
Models are saved to MongoDB for cloud-based serverless deployment
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
        
        # Define columns to EXCLUDE from features
        # IMPORTANT: Exclude pollutant concentrations (pm25, pm10, etc.) as they have 
        # near-perfect correlation with AQI (AQI is calculated from these values)
        # This would cause data leakage and artificially high R² scores
        self.exclude_cols = [
            'aqi',              # Target variable
            'timestamp',        # Metadata
            'date',             # Metadata
            '_id',              # MongoDB ID
            'inserted_at',      # Metadata
            'station_name',     # Metadata
            'latitude',         # Static location
            'longitude',        # Static location
            # Pollutant concentrations - EXCLUDE to prevent data leakage
            'pm25',             # Highly correlated with AQI (r=0.97)
            'pm10',             # Highly correlated with AQI
            'o3',               # Component of AQI calculation
            'no2',              # Component of AQI calculation
            'so2',              # Component of AQI calculation
            'co',               # Component of AQI calculation
        ]
        
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
        print(f"\n=== Preparing Data ===")
        print(f"Total records: {len(df)}")
        
        # Sort by timestamp first
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add lag features for AQI (previous AQI values are legitimate predictors)
        print("Adding AQI lag features...")
        lag_periods = [1, 3, 6, 12, 24]  # Hours ago
        for lag in lag_periods:
            df[f'aqi_lag_{lag}h'] = df[target_col].shift(lag)
        
        # Add rolling statistics
        df['aqi_rolling_mean_6h'] = df[target_col].rolling(window=6, min_periods=1).mean().shift(1)
        df['aqi_rolling_mean_12h'] = df[target_col].rolling(window=12, min_periods=1).mean().shift(1)
        df['aqi_rolling_mean_24h'] = df[target_col].rolling(window=24, min_periods=1).mean().shift(1)
        df['aqi_rolling_std_24h'] = df[target_col].rolling(window=24, min_periods=1).std().shift(1)
        
        # Add change rate
        df['aqi_change_1h'] = df[target_col].diff(1).shift(1)
        df['aqi_change_6h'] = df[target_col].diff(6).shift(1)
        
        # Remove first 24 rows (no lag data available)
        df = df.iloc[24:].copy()
        
        # Remove records with missing target
        df = df[df[target_col].notna()].copy()
        print(f"Records with valid AQI after lag processing: {len(df)}")
        
        # Select feature columns (exclude target and metadata)
        feature_cols = [col for col in df.columns if col not in self.exclude_cols]
        
        # Filter to only numeric columns with sufficient data
        valid_features = []
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                if df[col].notna().sum() / len(df) > 0.5:  # Keep if >50% valid
                    valid_features.append(col)
        
        if not valid_features:
            print("⚠ No valid features found, using default features")
            valid_features = [col for col in feature_cols if col in df.columns]
        
        print(f"Selected {len(valid_features)} features:")
        print(f"  {valid_features}")
        
        # Prepare X and y
        X = df[valid_features].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # Check if we have enough samples
        if len(X) < 20:
            print(f"Warning: Only {len(X)} samples available")
        
        # Split data (80% train, 20% test)
        test_size = max(0.2, min(0.3, 10/len(X)))  # Adjust for small datasets
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to keep feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=valid_features, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=valid_features, index=X_test.index)
        
        self.feature_names = valid_features
        
        return X_train_scaled, X_test_scaled, y_train, y_test, valid_features
    
    def train_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest model"""
        print(f"\n=== Training Random Forest ===")
        
        # Reduced n_estimators and max_depth for smaller model size (cloud storage)
        model = RandomForestRegressor(
            n_estimators=50,       # Reduced from 100 for smaller model
            max_depth=10,          # Reduced from 15 for smaller model
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
        print(f"  R²:   {results['test_r2']:.4f}")
        print(f"  MAE:  {results['test_mae']:.2f}")
        print(f"  RMSE: {results['test_rmse']:.2f}")
        
        return results
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Gradient Boosting model"""
        print(f"\n=== Training Gradient Boosting ===")
        
        # Optimized for smaller model size (cloud storage)
        model = GradientBoostingRegressor(
            n_estimators=80,       # Reduced slightly
            learning_rate=0.1,
            max_depth=4,           # Reduced from 5
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
        print(f"  R²:   {results['test_r2']:.4f}")
        print(f"  MAE:  {results['test_mae']:.2f}")
        print(f"  RMSE: {results['test_rmse']:.2f}")
        
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
        print(f"  R²:   {results['test_r2']:.4f}")
        print(f"  MAE:  {results['test_mae']:.2f}")
        print(f"  RMSE: {results['test_rmse']:.2f}")
        
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
        registry = []
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        registry = json.loads(content)
            except (json.JSONDecodeError, IOError):
                registry = []
        
        # Add new results
        results_copy = {}
        for model_name, result in self.results.items():
            result_copy = {}
            for key, value in result.items():
                if key == 'predictions':
                    # Convert numpy array to list
                    result_copy[key] = value.tolist() if hasattr(value, 'tolist') else list(value)
                else:
                    result_copy[key] = value
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

    def train_all_models(self, df: pd.DataFrame, db_handler=None) -> Dict:
        """
        Train all models and save to MongoDB (Cloud Storage)

        Args:
            df: Training data DataFrame
            db_handler: MongoDBHandler for cloud storage (REQUIRED)

        Returns:
            Dictionary with model results keyed by model name
        """
        # Prepare data
        X_train, X_test, y_train, y_test, feature_names = self.prepare_data(df)

        # Train models
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.train_ridge(X_train, y_train, X_test, y_test)

        # Compare and find best
        best_model_name = self.compare_models()

        # Save to MongoDB (PRIMARY STORAGE - Cloud)
        if db_handler:
            print(f"\n{'='*60}")
            print("SAVING ALL MODELS TO MONGODB CLOUD")
            print(f"{'='*60}")
            self.save_models_to_mongodb(db_handler, feature_names, best_model_name)
        else:
            print(f"\n⚠️ WARNING: No MongoDB handler provided!")
            print("Models were NOT saved to cloud storage.")
            print("This is not recommended for production.")
    
    def save_models_to_mongodb(self, db_handler, feature_names: List[str], best_model_name: str):
        """
        Save all trained models to MongoDB for cloud deployment
        Models are versioned and archived - never overwritten

        Args:
            db_handler: MongoDBHandler instance
            feature_names: List of feature names
            best_model_name: Name of the best performing model
        """
        print(f"\n{'='*60}")
        print("SAVING MODELS TO MONGODB CLOUD (Versioned Storage)")
        print(f"{'='*60}")

        # Serialize scaler once
        scaler_binary = pickle.dumps(self.scaler)

        models_data = []

        for model_name, model in self.models.items():
            # Serialize model
            model_binary = pickle.dumps(model)

            # Get metrics (without predictions array to save space)
            metrics = {
                'train_rmse': self.results[model_name]['train_rmse'],
                'train_mae': self.results[model_name]['train_mae'],
                'train_r2': self.results[model_name]['train_r2'],
                'test_rmse': self.results[model_name]['test_rmse'],
                'test_mae': self.results[model_name]['test_mae'],
                'test_r2': self.results[model_name]['test_r2'],
                'model_display_name': self.results[model_name]['model_name'],
                'timestamp': self.results[model_name]['timestamp']
            }

            models_data.append({
                'model_name': model_name,
                'model_binary': model_binary,
                'scaler_binary': scaler_binary,
                'feature_names': feature_names,
                'metrics': metrics,
                'is_best': (model_name == best_model_name)
            })

            print(f"  📦 Prepared: {model_name} (size: {len(model_binary)/1024:.1f} KB)")

        # Save all to MongoDB (versioned, never overwrites)
        success = db_handler.save_all_models(models_data)

        if success:
            print(f"\n✅ SUCCESS: All {len(models_data)} models saved to MongoDB Cloud!")
            print(f"   📦 Models collection: Latest versions")
            print(f"   📚 Models_archive collection: All versions preserved")
            print(f"   📜 Training_history collection: Training runs logged")
            print(f"   ⭐ Best model: {best_model_name}")
            print(f"\n   All models are safely stored and versioned in the cloud.")
        else:
            print(f"\n❌ FAILED: Could not save models to MongoDB")
            print(f"   Please check MongoDB connection and permissions.")
if __name__ == "__main__":
    print("Model trainer module loaded successfully")