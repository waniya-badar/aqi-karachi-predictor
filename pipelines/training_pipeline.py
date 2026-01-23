"""
Training Pipeline - Trains models daily using collected data
This pipeline fetches data from MongoDB, trains models, and saves them
Enhanced for CI/CD automation
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mongodb_handler import MongoDBHandler
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training_pipeline(min_days: int = 7, data_days: int = 120):
    """
    Main training pipeline function
    
    Args:
        min_days: Minimum days of data required for training
        data_days: Number of days of historical data to use
    
    Returns:
        bool: True if training successful, False otherwise
    """
    logger.info("Training Pipeline Started")
    
    start_time = datetime.utcnow()
    pipeline_status = {
        'start_time': start_time.isoformat(),
        'steps': {},
        'models': {},
        'success': False,
        'error': None
    }
    
    # Initialize components
    try:
        db_handler = MongoDBHandler()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        pipeline_status['error'] = str(e)
        _save_training_log(pipeline_status)
        return False
    
    try:
        # Step 1: Check data availability
        logger.info("Step 1: Checking data availability...")
        total_records = db_handler.db.features.count_documents({})
        
        logger.info(f"Total records available: {total_records}")
        
        if total_records < 100:
            logger.error(f"Insufficient data for training ({total_records} < 100)")
            pipeline_status['steps']['data_check'] = 'FAILED'
            pipeline_status['error'] = f"Only {total_records} records available (need ≥100)"
            _save_training_log(pipeline_status)
            return False
        
        pipeline_status['steps']['data_check'] = 'SUCCESS'
        pipeline_status['total_records'] = total_records
        logger.info("Data validation passed")
        
        # Step 2: Fetch training data
        logger.info(f"Step 2: Fetching {data_days} days of training data...")
        df = db_handler.get_training_data(days=data_days)
        
        if df is None or len(df) == 0:
            logger.error("Failed to fetch training data")
            pipeline_status['steps']['data_fetch'] = 'FAILED'
            pipeline_status['error'] = "No training data available"
            _save_training_log(pipeline_status)
            return False
        
        logger.info(f"Fetched {len(df)} records for training")
        pipeline_status['steps']['data_fetch'] = 'SUCCESS'
        pipeline_status['training_records'] = len(df)
        
        # Step 3: Train models (with MongoDB storage)
        logger.info("Step 3: Training models and saving to MongoDB...")
        trained_models = trainer.train_all_models(df, db_handler=db_handler)
        
        if not trained_models or len(trained_models) == 0:
            logger.error("Failed to train models")
            pipeline_status['steps']['training'] = 'FAILED'
            pipeline_status['error'] = "Model training failed"
            _save_training_log(pipeline_status)
            return False
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        pipeline_status['steps']['training'] = 'SUCCESS'
        pipeline_status['models_trained'] = len(trained_models)
        
        # Step 4: Verify models in MongoDB
        logger.info("Step 4: Verifying models in MongoDB...")
        try:
            stored_models = db_handler.get_all_models_metadata()
            best_model = db_handler.get_best_model()
            
            if stored_models:
                logger.info(f"Verified {len(stored_models)} models in MongoDB:")
                for m in stored_models:
                    metrics = m.get('metrics', {})
                    is_best = " (BEST)" if m.get('is_best') else ""
                    r2 = metrics.get('test_r2', 0)
                    mae = metrics.get('test_mae', 0)
                    rmse = metrics.get('test_rmse', 0)
                    logger.info(f"  - {m['model_name']}: R²={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}{is_best}")
                    pipeline_status['models'][m['model_name']] = metrics
                
                if best_model:
                    pipeline_status['best_model'] = best_model['model_name']
                
                pipeline_status['steps']['verification'] = 'SUCCESS'
            else:
                logger.warning("No models found in MongoDB after training")
                pipeline_status['steps']['verification'] = 'WARNING'
                
        except Exception as e:
            logger.warning(f"Failed to verify models: {e}")
            pipeline_status['steps']['verification'] = 'WARNING'
        
        pipeline_status['success'] = True
        pipeline_status['end_time'] = datetime.utcnow().isoformat()
        
        logger.info("Training Pipeline Completed Successfully")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error in training pipeline: {e}", exc_info=True)
        pipeline_status['error'] = str(e)
        pipeline_status['end_time'] = datetime.utcnow().isoformat()
        return False
    
    finally:
        try:
            db_handler.close()
        except:
            pass


if __name__ == "__main__":
    # Run training pipeline with 3+ months of data
    run_training_pipeline(min_days=7, data_days=90)