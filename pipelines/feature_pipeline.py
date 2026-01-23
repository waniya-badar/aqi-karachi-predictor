"""
Feature Pipeline - Runs every hour to collect and store data
"""

import sys
import os
import logging
from datetime import datetime
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_fetcher import AQICNFetcher
from src.feature_engineering import FeatureEngineer
from src.mongodb_handler import MongoDBHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_feature_pipeline(dry_run=False):
    """Main feature pipeline function"""
    logger.info("Feature Pipeline Started")
    
    start_time = datetime.utcnow()
    pipeline_status = {
        'start_time': start_time.isoformat(),
        'steps': {},
        'success': False,
        'error': None
    }
    
    db_handler = None
    
    try:
        fetcher = AQICNFetcher()
        engineer = FeatureEngineer()
        db_handler = MongoDBHandler()
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        pipeline_status['error'] = str(e)
        _save_pipeline_log(pipeline_status)
        return False
    
    try:
        logger.info("Step 1: Fetching current AQI data...")
        raw_data = fetcher.fetch_current_data()
        
        if not raw_data:
            logger.error("Failed to fetch data from API")
            pipeline_status['steps']['fetch'] = 'FAILED'
            _save_pipeline_log(pipeline_status)
            return False
        
        logger.info(f"Successfully fetched AQI data: {raw_data.get('aqi', 'N/A')}")
        pipeline_status['steps']['fetch'] = 'SUCCESS'
        
        logger.info("Step 2: Engineering features...")
        features = engineer.create_features(raw_data)
        
        if not features:
            logger.error("Failed to create features - validation failed or missing data")
            pipeline_status['steps']['engineering'] = 'FAILED'
            _save_pipeline_log(pipeline_status)
            return False
        
        # Additional validation: ensure all critical fields are present and non-null
        required_fields = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 
                         'temperature', 'humidity', 'pressure', 'wind_speed']
        missing = [f for f in required_fields if features.get(f) in (None, '', 'null')]
        if missing:
            logger.error(f"Validation failed: Missing fields {missing}")
            pipeline_status['steps']['engineering'] = 'FAILED'
            _save_pipeline_log(pipeline_status)
            return False
        
        logger.info(f"Successfully created {len(features)} features")
        pipeline_status['steps']['engineering'] = 'SUCCESS'
        
        if not dry_run:
            logger.info("Step 3: Storing features in MongoDB...")
            success = db_handler.insert_features(features)
            
            if not success:
                logger.error("Failed to store features")
                pipeline_status['steps']['storage'] = 'FAILED'
                _save_pipeline_log(pipeline_status)
                return False
            
            logger.info("Successfully stored features in MongoDB")
            pipeline_status['steps']['storage'] = 'SUCCESS'
            
            count = db_handler.db.features.count_documents({})
            logger.info(f"Total records: {count}")
            pipeline_status['total_records'] = count
        
        pipeline_status['success'] = True
        pipeline_status['end_time'] = datetime.utcnow().isoformat()
        
        print(f"\nFeature Pipeline Completed Successfully!")
        print(f"AQI: {raw_data.get('aqi', 'N/A')}, Features: {len(features)}\n")
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        pipeline_status['error'] = str(e)
        return False
        
    finally:
        if db_handler:
            try:
                db_handler.close()
            except:
                pass


if __name__ == "__main__":
    run_feature_pipeline()
