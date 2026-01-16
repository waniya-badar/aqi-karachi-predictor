"""
Model Explainer - Uses SHAP for feature importance
Explains which features most affect AQI predictions
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Optional


class ModelExplainer:
    """Explains model predictions using SHAP values"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize explainer with trained model
        
        Args:
            model_path: Path to pickled model file
        """
        self.model = None
        self.explainer = None
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f" Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f" Error loading model: {e}")
            return False
    
    def create_explainer(self, X_train: pd.DataFrame):
        """
        Create SHAP explainer
        
        Args:
            X_train: Training data for background samples
        """
        try:
            print("Creating SHAP explainer...")
            
            # For tree-based models, use TreeExplainer (faster)
            if hasattr(self.model, 'estimators_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For other models, use KernelExplainer with sample
                background = shap.sample(X_train, min(100, len(X_train)))
                self.explainer = shap.KernelExplainer(self.model.predict, background)
            
            self.feature_names = X_train.columns.tolist()
            print(" SHAP explainer created")
            return True
            
        except Exception as e:
            print(f" Error creating explainer: {e}")
            return False
    
    def calculate_shap_values(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for dataset
        
        Args:
            X: Feature dataset
        
        Returns:
            SHAP values array
        """
        try:
            print(f"Calculating SHAP values for {len(X)} samples...")
            shap_values = self.explainer.shap_values(X)
            print(" SHAP values calculated")
            return shap_values
        except Exception as e:
            print(f" Error calculating SHAP values: {e}")
            return None
    
    def plot_feature_importance(self, X: pd.DataFrame, save_path: str = None):
        """
        Plot feature importance using SHAP
        
        Args:
            X: Feature dataset
            save_path: Path to save plot
        """
        shap_values = self.calculate_shap_values(X)
        
        if shap_values is None:
            return
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP Values)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Plot saved to {save_path}")
        
        plt.show()
    
    def plot_summary(self, X: pd.DataFrame, save_path: str = None):
        """
        Plot SHAP summary (beeswarm plot)
        
        Args:
            X: Feature dataset
            save_path: Path to save plot
        """
        shap_values = self.calculate_shap_values(X)
        
        if shap_values is None:
            return
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP Summary Plot", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def explain_prediction(self, X_single: pd.DataFrame, save_path: str = None):
        """
        Explain a single prediction with waterfall plot
        
        Args:
            X_single: Single row DataFrame
            save_path: Path to save plot
        """
        shap_values = self.calculate_shap_values(X_single)
        
        if shap_values is None:
            return
        
        # Create explanation object for waterfall plot
        expected_value = self.explainer.expected_value
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=expected_value,
                data=X_single.iloc[0].values,
                feature_names=X_single.columns.tolist()
            ),
            show=False
        )
        plt.title("SHAP Explanation for Single Prediction", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Plot saved to {save_path}")
        
        plt.show()
    
    def get_top_features(self, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            X: Feature dataset
            top_n: Number of top features
        
        Returns:
            DataFrame with feature importance scores
        """
        shap_values = self.calculate_shap_values(X)
        
        if shap_values is None:
            return None
        
        # Calculate mean absolute SHAP value for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("="*50)
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:.<40} {row['importance']:.4f}")
        
        return importance_df.head(top_n)
    
    def generate_explanation_report(self, X_test: pd.DataFrame, 
                                   output_dir: str = 'models/explanations'):
        """
        Generate complete explanation report with all plots
        
        Args:
            X_test: Test dataset
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating SHAP Explanation Report")
        print("="*60 + "\n")
        
        # 1. Feature Importance Bar Plot
        print("1. Creating feature importance plot...")
        self.plot_feature_importance(
            X_test, 
            save_path=os.path.join(output_dir, 'feature_importance.png')
        )
        
        # 2. Summary Plot
        print("\n2. Creating summary plot...")
        self.plot_summary(
            X_test,
            save_path=os.path.join(output_dir, 'shap_summary.png')
        )
        
        # 3. Top Features Table
        print("\n3. Analyzing top features...")
        top_features = self.get_top_features(X_test, top_n=15)
        top_features.to_csv(
            os.path.join(output_dir, 'top_features.csv'),
            index=False
        )
        print(f" Top features saved to {output_dir}/top_features.csv")
        
        # 4. Single Prediction Example
        print("\n4. Creating example prediction explanation...")
        self.explain_prediction(
            X_test.iloc[[0]],
            save_path=os.path.join(output_dir, 'example_prediction.png')
        )
        
        print("\n" + "="*60)
        print(f" Complete explanation report saved to {output_dir}/")
        print("="*60 + "\n")


def run_explainability_analysis():
    """Run complete explainability analysis"""
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from src.mongodb_handler import MongoDBHandler
    from src.feature_engineering import FeatureEngineer
    from src.model_trainer import ModelTrainer
    
    print("\n" + "="*60)
    print("Model Explainability Analysis")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data from MongoDB...")
    db_handler = MongoDBHandler()
    df = db_handler.get_training_data(days=30)  # Last 30 days
    
    if df is None or len(df) < 100:
        print(" Insufficient data for analysis")
        return
    
    print(f" Loaded {len(df)} records")
    
    # Prepare features
    print("\nPreparing features...")
    engineer = FeatureEngineer()
    df = engineer.add_lag_features(df)
    df = engineer.handle_missing_values(df)
    
    # Prepare training data
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(df)
    
    print(f" Training set: {len(X_train)} samples")
    print(f" Test set: {len(X_test)} samples")
    
    # Load best model (try all three)
    model_names = ['gradient_boosting', 'random_forest', 'ridge']
    
    for model_name in model_names:
        model_path = f'models/saved_models/{model_name}_latest.pkl'
        
        if not os.path.exists(model_path):
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name.replace('_', ' ').title()} Model")
        print(f"{'='*60}\n")
        
        # Create explainer
        explainer = ModelExplainer(model_path)
        explainer.create_explainer(X_train)
        
        # Generate report
        output_dir = f'models/explanations/{model_name}'
        explainer.generate_explanation_report(X_test, output_dir)
    
    db_handler.close()
    print("\n✓ Explainability analysis complete!")


if __name__ == "__main__":
    run_explainability_analysis()