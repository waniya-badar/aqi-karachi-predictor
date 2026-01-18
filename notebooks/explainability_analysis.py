#!/usr/bin/env python3
"""
Comprehensive Explainability Analysis using SHAP and LIME
Analyzes feature importance for AQI prediction models
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
import joblib
import json

# ML Libraries
import shap
import lime
import lime.lime_tabular

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mongodb_handler import MongoDBHandler
from src.feature_engineering import FeatureEngineer

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_models():
    """Load all trained models"""
    print("Loading trained models...")
    
    models = {}
    model_dir = 'models/saved_models'
    
    try:
        models['ridge'] = joblib.load(f'{model_dir}/ridge_latest.pkl')
        print("[OK] Loaded Ridge Regression model")
    except Exception as e:
        print(f"[FAIL] Could not load Ridge model: {e}")
    
    try:
        models['gradient_boosting'] = joblib.load(f'{model_dir}/gradient_boosting_latest.pkl')
        print("[OK] Loaded Gradient Boosting model")
    except Exception as e:
        print(f"[FAIL] Could not load Gradient Boosting model: {e}")
    
    try:
        models['random_forest'] = joblib.load(f'{model_dir}/random_forest_latest.pkl')
        print("[OK] Loaded Random Forest model")
    except Exception as e:
        print(f"[FAIL] Could not load Random Forest model: {e}")
    
    return models


def load_feature_names():
    """Load feature names from saved file"""
    try:
        with open('models/saved_models/feature_names.json', 'r') as f:
            feature_data = json.load(f)
            if isinstance(feature_data, dict) and 'features' in feature_data:
                return feature_data['features']
            return feature_data
    except:
        return None


def load_data():
    """Load data from MongoDB"""
    print("\nLoading data from MongoDB...")
    
    try:
        db_handler = MongoDBHandler()
        
        # Fetch 60 days of data
        df = db_handler.get_training_data(days=60)
        
        if df is None:
            print("[FAIL] Could not fetch data")
            return None
        
        print(f"[OK] Loaded {len(df)} records")
        db_handler.close()
        
        return df
    
    except Exception as e:
        print(f"[FAIL] Error loading data: {e}")
        return None


def prepare_features(df):
    """Prepare features for model prediction"""
    print("\nPreparing features...")
    
    try:
        engineer = FeatureEngineer()
        
        # Extract features from each row
        X_list = []
        for idx, row in df.iterrows():
            raw_data = {
                'timestamp': row['timestamp'],
                'pm25': row.get('pm25', 0),
                'pm10': row.get('pm10', 0),
                'no2': row.get('no2', 0),
                'so2': row.get('so2', 0),
                'co': row.get('co', 0),
                'o3': row.get('o3', 0),
                'temperature': row.get('temperature', 0),
                'humidity': row.get('humidity', 0),
                'wind_speed': row.get('wind_speed', 0),
                'pressure': row.get('pressure', 0),
                'visibility': row.get('visibility', 0)
            }
            features = engineer.create_features(raw_data)
            if features:
                X_list.append(list(features.values()))
        
        # Convert to DataFrame
        feature_names = load_feature_names()
        if feature_names and len(X_list) > 0:
            X = pd.DataFrame(X_list, columns=feature_names[:len(X_list[0])])
        else:
            X = pd.DataFrame(X_list)
        
        print(f"[OK] Prepared {len(X)} samples with {X.shape[1]} features")
        
        return X, feature_names if feature_names else list(X.columns)
    
    except Exception as e:
        print(f"[FAIL] Error preparing features: {e}")
        return None, None


def explain_with_shap(model, X, feature_names, model_name):
    """Generate SHAP explanations"""
    print(f"\nSHAP Analysis - {model_name}\n")
    
    try:
        # Sample data for SHAP
        background_data = X.iloc[:min(100, len(X))]
        
        # Create SHAP explainer
        if 'TreeExplainer' in str(type(model)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X.iloc[:min(100, len(X))])
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features (SHAP):")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']:.<40} {row['Importance']:.4f}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Feature importance bar plot
        axes[0].barh(importance_df['Feature'].iloc[:15], importance_df['Importance'].iloc[:15], color='steelblue')
        axes[0].set_xlabel('Mean |SHAP value|')
        axes[0].set_title(f'SHAP Feature Importance - {model_name}')
        axes[0].invert_yaxis()
        
        # SHAP summary plot
        top_features_idx = importance_df.head(10).index
        top_features = importance_df.loc[top_features_idx, 'Feature'].tolist()
        top_feature_indices = [list(feature_names).index(f) for f in top_features]
        
        axes[1].scatter(shap_values[:, top_feature_indices[0]], 
                       np.random.normal(0, 0.1, len(shap_values)), alpha=0.5)
        axes[1].set_xlabel(f"SHAP value for {top_features[0]}")
        axes[1].set_ylabel("Impact")
        axes[1].set_title(f'SHAP Values Distribution - {model_name}')
        
        plt.tight_layout()
        plt.savefig(f'notebooks/plots/shap_analysis_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        print(f"\nSHAP plot saved: shap_analysis_{model_name.lower().replace(' ', '_')}.png")
        plt.close()
        
        return importance_df
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return None


def explain_with_lime(model, X, feature_names, model_name):
    """Generate LIME explanations"""
    print(f"\nLIME Analysis - {model_name}\n")
    
    try:
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values, 
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        
        # Explain multiple instances
        num_samples = min(5, len(X))
        lime_importances = pd.DataFrame(columns=['Feature', 'Weight'])
        
        for i in range(num_samples):
            exp = explainer.explain_instance(X.iloc[i], model.predict, num_features=15)
            exp_list = exp.as_list()
            
            for feature, weight in exp_list:
                # Extract feature name
                feature_name = feature.split()[0] if feature else 'unknown'
                if feature_name in feature_names:
                    lime_importances = pd.concat([
                        lime_importances,
                        pd.DataFrame({'Feature': [feature_name], 'Weight': [abs(weight)]})
                    ], ignore_index=True)
        
        # Aggregate LIME importances
        if len(lime_importances) > 0:
            lime_agg = lime_importances.groupby('Feature')['Weight'].mean().sort_values(ascending=False)
            
            print("\nTop 10 Most Important Features (LIME):")
            for idx, (feature, weight) in enumerate(lime_agg.head(10).items()):
                print(f"  {feature:.<40} {weight:.4f}")
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            lime_agg.head(15).plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Mean Weight')
            ax.set_title(f'LIME Feature Importance - {model_name}')
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f'notebooks/plots/lime_analysis_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
            print(f"\nLIME plot saved: lime_analysis_{model_name.lower().replace(' ', '_')}.png")
            plt.close()
            
            return lime_agg.to_frame(name='Importance').reset_index()
        else:
            print("No LIME importances calculated")
            return None
        
    except Exception as e:
        print(f"Error in LIME analysis: {e}")
        return None


def create_comparison_plot(all_importances, models_list):
    """Create comparison plot of feature importances across models"""
    print(f"\nCOMPARISON: Feature Importance Across All Models\n")
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        for idx, (model_name, importance_df) in enumerate(all_importances.items()):
            if importance_df is not None:
                top_n = min(12, len(importance_df))
                importance_df_top = importance_df.head(top_n)
                
                if 'Importance' in importance_df_top.columns:
                    axes[idx].barh(importance_df_top['Feature'], importance_df_top['Importance'], color=f'C{idx}')
                else:
                    axes[idx].barh(importance_df_top['Feature'], importance_df_top.iloc[:, 1], color=f'C{idx}')
                
                axes[idx].set_xlabel('Importance Score')
                axes[idx].set_title(f'{model_name}')
                axes[idx].invert_yaxis()
        
        plt.suptitle('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('notebooks/plots/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print("[OK] Comparison plot saved: feature_importance_comparison.png")
        plt.close()
        
    except Exception as e:
        print(f"[FAIL] Error creating comparison plot: {e}")


def main():
    """Main function"""
    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSIS - SHAP & LIME")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_data()
    if df is None:
        print("[FAIL] Cannot proceed without data")
        return False
    
    # Prepare features
    X, feature_names = prepare_features(df)
    if X is None or feature_names is None:
        print("[FAIL] Cannot proceed without features")
        return False
    
    # Extract target variable
    y = df['aqi'].values if 'aqi' in df.columns else None
    if y is None:
        print("No target variable (AQI) found")
        return False
    
    models = load_models()
    if not models:
        print("Cannot proceed without models")
        return False
    
    os.makedirs('notebooks/plots', exist_ok=True)
    
    all_importances = {}
    
    for model_name, model in models.items():
        if model is None:
            continue
        
        display_name = model_name.replace('_', ' ').title()
        
        shap_importance = explain_with_shap(model, X, feature_names, display_name)
        if shap_importance is not None:
            all_importances[f"{display_name} (SHAP)"] = shap_importance
        
        lime_importance = explain_with_lime(model, X, feature_names, display_name)
        if lime_importance is not None:
            all_importances[f"{display_name} (LIME)"] = lime_importance
    
    if all_importances:
        create_comparison_plot(all_importances, list(models.keys()))
    
    print(f"\nEXPLAINABILITY ANALYSIS COMPLETE\n")
    print("Generated files:")
    print("  - notebooks/plots/shap_analysis_*.png")
    print("  - notebooks/plots/lime_analysis_*.png")
    print("  - notebooks/plots/feature_importance_comparison.png")
    print("\nAll analysis complete!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
