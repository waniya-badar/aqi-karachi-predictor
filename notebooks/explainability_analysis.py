#!/usr/bin/env python3
"""
Comprehensive Explainability Analysis using SHAP and LIME
Loads models from MongoDB cloud and analyzes feature importance
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ML Libraries
import shap
import lime
import lime.lime_tabular

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mongodb_handler import MongoDBHandler
from src.model_trainer import ModelTrainer

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create plots directory
os.makedirs('notebooks/plots', exist_ok=True)


def load_models_from_mongodb():
    """Load all trained models from MongoDB cloud"""
    print("\nLoading trained models from MongoDB...")
    
    db_handler = MongoDBHandler()
    models = {}
    scalers = {}
    feature_names = None
    
    model_names = ['random_forest', 'gradient_boosting', 'ridge']
    
    for model_name in model_names:
        model_data = db_handler.get_model(model_name)
        if model_data:
            import pickle
            models[model_name] = pickle.loads(model_data['model_binary'])
            scalers[model_name] = pickle.loads(model_data['scaler_binary'])
            if feature_names is None:
                feature_names = model_data.get('feature_names', [])
            metrics = model_data.get('metrics', {})
            r2 = metrics.get('test_r2', 0)
            mae = metrics.get('test_mae', 0)
            rmse = metrics.get('test_rmse', 0)
            print(f"  [OK] Loaded {model_name} (R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f})")
        else:
            print(f"  [FAIL] Could not load {model_name}")
    
    db_handler.close()
    return models, scalers, feature_names


def load_data():
    """Load all data from MongoDB"""
    print("\nLoading data from MongoDB...")
    
    try:
        db_handler = MongoDBHandler()
        df = db_handler.get_training_data(days=365)  # Get up to 1 year of data
        
        if df is None or len(df) == 0:
            print("[FAIL] Could not fetch data")
            return None
        
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days
        print(f"  [OK] Loaded {len(df)} records ({date_range} days of data)")
        db_handler.close()
        
        return df
    
    except Exception as e:
        print(f"[FAIL] Error loading data: {e}")
        return None


def prepare_features(df, feature_names):
    """Prepare features for model prediction - matches training data preparation"""
    print("\nPreparing features...")
    
    try:
        # First add ALL AQI lag features (same as training)
        df = df.sort_values('timestamp').copy()
        
        # Basic lag features
        df['aqi_lag_1h'] = df['aqi'].shift(1)
        df['aqi_lag_3h'] = df['aqi'].shift(3)
        df['aqi_lag_6h'] = df['aqi'].shift(6)
        df['aqi_lag_12h'] = df['aqi'].shift(12)
        df['aqi_lag_24h'] = df['aqi'].shift(24)
        
        # Rolling statistics
        df['aqi_rolling_mean_6h'] = df['aqi'].rolling(window=6, min_periods=1).mean()
        df['aqi_rolling_mean_12h'] = df['aqi'].rolling(window=12, min_periods=1).mean()
        df['aqi_rolling_mean_24h'] = df['aqi'].rolling(window=24, min_periods=1).mean()
        df['aqi_rolling_std_24h'] = df['aqi'].rolling(window=24, min_periods=1).std()
        
        # Change features
        df['aqi_change_1h'] = df['aqi'] - df['aqi_lag_1h']
        df['aqi_change_6h'] = df['aqi'] - df['aqi_lag_6h']
        df['aqi_change_24h'] = df['aqi'] - df['aqi_lag_24h']
        
        # Drop rows with NaN from lag features
        df = df.dropna(subset=['aqi_lag_24h'])
        
        if feature_names:
            # Use only the features the model expects
            available_features = [f for f in feature_names if f in df.columns]
            missing_features = [f for f in feature_names if f not in df.columns]
            
            if missing_features:
                print(f"  [WARN] Missing features: {missing_features}")
            
            if not available_features:
                print(f"  [FAIL] No matching features found")
                return None, None, None
        else:
            print("  [FAIL] No feature names provided")
            return None, None, None
        
        print(f"  Using {len(available_features)} features")
        
        X = df[available_features].copy()
        X = X.fillna(X.median(numeric_only=True))
        y = df['aqi'].values
        
        print(f"  [OK] Prepared {len(X)} samples with {X.shape[1]} features")
        
        return X, available_features, y
    
    except Exception as e:
        print(f"[FAIL] Error preparing features: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def explain_with_shap(model, X, feature_names, model_name, scaler=None):
    """Generate SHAP explanations"""
    print(f"\n{'='*50}")
    print(f"SHAP Analysis - {model_name}")
    print(f"{'='*50}")
    
    try:
        # Sample data for SHAP (use more samples for better analysis)
        n_samples = min(200, len(X))
        X_sample = X.iloc[:n_samples].copy()
        
        # Scale if scaler is available
        if scaler is not None:
            X_scaled = pd.DataFrame(scaler.transform(X_sample), columns=feature_names)
        else:
            X_scaled = X_sample
        
        # Create SHAP explainer based on model type
        model_type = type(model).__name__
        
        if 'Forest' in model_type or 'Gradient' in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        else:
            # For linear models like Ridge
            background = X_scaled.iloc[:min(50, len(X_scaled))]
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_scaled.iloc[:min(50, len(X_scaled))])
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features (SHAP):")
        for idx, row in importance_df.head(10).iterrows():
            bar = '#' * int(row['Importance'] / importance_df['Importance'].max() * 20)
            print(f"  {row['Feature']:<25} {row['Importance']:>8.4f}  {bar}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Feature importance bar plot
        top_n = min(15, len(importance_df))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
        axes[0].barh(importance_df['Feature'].iloc[:top_n], 
                     importance_df['Importance'].iloc[:top_n], 
                     color=colors)
        axes[0].set_xlabel('Mean |SHAP value|', fontsize=12)
        axes[0].set_title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        
        # SHAP summary beeswarm-style plot (simplified)
        top_feature = importance_df['Feature'].iloc[0]
        top_idx = list(feature_names).index(top_feature)
        
        scatter_x = shap_values[:, top_idx]
        scatter_y = np.random.normal(0, 0.15, len(scatter_x))
        scatter_c = X_scaled[top_feature].values[:len(scatter_x)]
        
        scatter = axes[1].scatter(scatter_x, scatter_y, c=scatter_c, 
                                  cmap='coolwarm', alpha=0.6, s=30)
        axes[1].set_xlabel(f"SHAP value for {top_feature}", fontsize=12)
        axes[1].set_ylabel("Density", fontsize=12)
        axes[1].set_title(f'SHAP Values Distribution - Top Feature', fontsize=14, fontweight='bold')
        axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=axes[1], label='Feature Value')
        
        plt.tight_layout()
        filename = f'notebooks/plots/shap_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {filename}")
        plt.close()
        
        return importance_df
        
    except Exception as e:
        print(f"  Error in SHAP analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def explain_with_lime(model, X, feature_names, model_name, scaler=None):
    """Generate LIME explanations"""
    print(f"\n{'='*50}")
    print(f"LIME Analysis - {model_name}")
    print(f"{'='*50}")
    
    try:
        # Scale if scaler is available
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_scaled, 
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        
        # Explain multiple instances
        num_samples = min(10, len(X))
        lime_weights = {f: [] for f in feature_names}
        
        print(f"\n  Analyzing {num_samples} sample instances...")
        
        for i in range(num_samples):
            exp = explainer.explain_instance(
                X_scaled[i], 
                model.predict, 
                num_features=len(feature_names)
            )
            
            for feature_exp, weight in exp.as_list():
                # Parse feature name from LIME output
                for f in feature_names:
                    if f in feature_exp:
                        lime_weights[f].append(abs(weight))
                        break
        
        # Calculate average weights
        avg_weights = {f: np.mean(w) if w else 0 for f, w in lime_weights.items()}
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': list(avg_weights.keys()),
            'Importance': list(avg_weights.values())
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features (LIME):")
        for idx, row in importance_df.head(10).iterrows():
            bar = '#' * int(row['Importance'] / max(importance_df['Importance'].max(), 0.001) * 20)
            print(f"  {row['Feature']:<25} {row['Importance']:>8.4f}  {bar}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(15, len(importance_df))
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, top_n))
        
        importance_df_top = importance_df.head(top_n)
        ax.barh(importance_df_top['Feature'], importance_df_top['Importance'], color=colors)
        ax.set_xlabel('Mean Absolute Weight', fontsize=12)
        ax.set_title(f'LIME Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        filename = f'notebooks/plots/lime_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {filename}")
        plt.close()
        
        return importance_df
        
    except Exception as e:
        print(f"  Error in LIME analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_plot(all_importances):
    """Create comparison plot of feature importances across models"""
    print(f"\n{'='*50}")
    print("COMPARISON: Feature Importance Across All Models")
    print(f"{'='*50}")
    
    try:
        # Filter out None values
        valid_importances = {k: v for k, v in all_importances.items() if v is not None}
        
        if len(valid_importances) == 0:
            print("  No valid importance data to compare")
            return
        
        n_models = len(valid_importances)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        colors = ['steelblue', 'coral', 'forestgreen', 'purple', 'orange', 'crimson']
        
        for idx, (model_name, importance_df) in enumerate(valid_importances.items()):
            top_n = min(12, len(importance_df))
            importance_df_top = importance_df.head(top_n)
            
            axes[idx].barh(importance_df_top['Feature'], 
                          importance_df_top['Importance'], 
                          color=colors[idx % len(colors)],
                          alpha=0.8)
            axes[idx].set_xlabel('Importance Score', fontsize=11)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
        
        plt.suptitle('Feature Importance Comparison Across Models & Methods', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('notebooks/plots/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print("\n  Saved: notebooks/plots/feature_importance_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"  Error creating comparison plot: {e}")


def generate_summary_report(all_importances, df):
    """Generate a summary report of the explainability analysis"""
    print(f"\n{'='*70}")
    print("EXPLAINABILITY SUMMARY REPORT")
    print(f"{'='*70}")
    
    # Find most consistently important features across all methods
    feature_counts = {}
    
    for model_name, importance_df in all_importances.items():
        if importance_df is not None:
            top_features = importance_df.head(5)['Feature'].tolist()
            for f in top_features:
                feature_counts[f] = feature_counts.get(f, 0) + 1
    
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📊 Most Consistently Important Features (across all models/methods):")
    for feature, count in sorted_features[:10]:
        stars = '⭐' * count
        print(f"   {feature:<25} appears in top-5 of {count} analyses {stars}")
    
    print(f"\n📈 Dataset Summary:")
    print(f"   Total Records: {len(df)}")
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"   Data Span: {date_range} days (~{date_range/30:.1f} months)")
    print(f"   AQI Range: {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
    print(f"   Average AQI: {df['aqi'].mean():.1f}")
    
    print(f"\n💡 Key Insights:")
    if sorted_features:
        top_feature = sorted_features[0][0]
        print(f"   • '{top_feature}' is the most important predictor across models")
        
        lag_features = [f for f, c in sorted_features if 'lag' in f or 'rolling' in f]
        if lag_features:
            print(f"   • Temporal patterns (lag features) are highly predictive")
        
        weather_features = [f for f, c in sorted_features if f in ['temperature', 'humidity', 'wind_speed', 'pressure']]
        if weather_features:
            print(f"   • Weather features that matter: {', '.join(weather_features)}")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("EXPLAINABILITY ANALYSIS - SHAP & LIME")
    print("Models loaded from MongoDB Cloud")
    print("="*70)
    
    # Load data
    df = load_data()
    if df is None:
        print("\n[FAIL] Cannot proceed without data")
        return False
    
    # Load models from MongoDB
    models, scalers, feature_names = load_models_from_mongodb()
    if not models:
        print("\n[FAIL] Cannot proceed without models")
        return False
    
    if not feature_names:
        print("\n[FAIL] Cannot proceed without feature names")
        return False
    
    # Prepare features
    X, feature_names, y = prepare_features(df, feature_names)
    if X is None:
        print("\n[FAIL] Cannot proceed without prepared features")
        return False
    
    all_importances = {}
    
    # Analyze each model
    for model_name, model in models.items():
        if model is None:
            continue
        
        display_name = model_name.replace('_', ' ').title()
        scaler = scalers.get(model_name)
        
        # SHAP Analysis
        shap_importance = explain_with_shap(model, X, feature_names, display_name, scaler)
        if shap_importance is not None:
            all_importances[f"{display_name} (SHAP)"] = shap_importance
        
        # LIME Analysis
        lime_importance = explain_with_lime(model, X, feature_names, display_name, scaler)
        if lime_importance is not None:
            all_importances[f"{display_name} (LIME)"] = lime_importance
    
    # Create comparison plot
    if all_importances:
        create_comparison_plot(all_importances)
    
    # Generate summary report
    generate_summary_report(all_importances, df)
    
    print(f"\n{'='*70}")
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("\nGenerated files in notebooks/plots/:")
    print("  • shap_*.png - SHAP analysis for each model")
    print("  • lime_*.png - LIME analysis for each model")
    print("  • feature_importance_comparison.png - Cross-model comparison")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
