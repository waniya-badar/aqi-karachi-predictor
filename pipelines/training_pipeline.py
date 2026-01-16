"""
Training Pipeline - Trains models daily using collected data
This pipeline fetches data from MongoDB, trains models, and saves them
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mongodb_handler import MongoDBHandler
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer


def run_training_pipeline(min_days: int = 7):
    """
    Main training pipeline function
    
    Args:
        min_days: Minimum days of data required for training
    """
    print(f"\n{'='*60}")
    print(f"Training Pipeline Started: {datetime.now()}")
    print(f"{'='*60}\n")
    
    # Initialize components
    db_handler = MongoDBHandler()
    engineer = FeatureEngineer()
    trainer = ModelTrainer()
    
    try:
        #Check if we have enough data
        print("Step 1: Checking data availability...")
        stats = db_handler.get_data_statistics()
        
        total_records = stats['total_records']
        date_range_days = stats['date_range_days']
        
        print(f"Total records: {total_records}")
        print(f"Date range: {date_range_days} days")
        
        if total_records < 100:
            print(f"\nInsufficient data for training!")
            print(f"Need at least 100 records, have {total_records}")
            print(f"Please run feature pipeline hourly to collect more data")
            return False
        
        if date_range_days < min_days:
            print(f"\nInsufficient date range!")
            print(f"Need at least {min_days} days, have {date_range_days} days")
            print(f"Continue collecting data...")
            return False
        
        print(f"Sufficient data available for training")
        
        #Fetch training data
        print(f"\nStep 2: Fetching training data from MongoDB...")
        df = db_handler.get_training_data(days=120)  # Last 4 months
        
        if df is None or len(df) == 0:
            print("No data retrieved from database")
            return False
        
        print(f"Retrieved {len(df)} records")
        
        #Add lag features
        print(f"\nStep 3: Adding lag and rolling features...")
        df = engineer.add_lag_features(df)
        df = engineer.handle_missing_values(df)
        
        print(f"Feature engineering complete: {len(df)} records, {len(df.columns)} features")
        
        #Prepare data for training
        print(f"\nStep 4: Preparing data for training...")
        X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(df)
        
        #Train all models
        print(f"\nStep 5: Training models...")
        
        trainer.train_random_forest(X_train, y_train, X_test, y_test)
        trainer.train_gradient_boosting(X_train, y_train, X_test, y_test)
        trainer.train_ridge(X_train, y_train, X_test, y_test)
        
        #Compare models
        print(f"\nStep 6: Comparing model performance...")
        best_model = trainer.compare_models()
        
        #Save models
        print(f"\nStep 7: Saving models and metadata...")
        trainer.save_models(feature_names)
        
        #SHAP Explainability Analysis
        print(f"\nStep 8: Running SHAP explainability analysis...")
        try:
            from src.model_explainer import ModelExplainer
            
            explainer = ModelExplainer(f'models/saved_models/{best_model}_latest.pkl')
            explainer.create_explainer(X_train)
            explainer.generate_explanation_report(X_test, f'models/explanations/{best_model}')
            print("SHAP analysis complete")
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"Training Pipeline Completed Successfully!")
        print(f"Best Model: {trainer.results[best_model]['model_name']}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Always close database connection
        db_handler.close()


if __name__ == "__main__":
    # Run training pipeline
    run_training_pipeline(min_days=7)