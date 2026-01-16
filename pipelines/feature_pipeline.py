"""
Feature Pipeline - Runs every hour to collect and store data
This is the main pipeline that fetches data and stores features
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_fetcher import AQICNFetcher
from src.feature_engineering import FeatureEngineer
from src.mongodb_handler import MongoDBHandler


def run_feature_pipeline():
    """
    Main feature pipeline function
    Fetches current AQI data, creates features, and stores in MongoDB
    """
    print(f"\n{'='*60}")
    print(f"Feature Pipeline Started: {datetime.now()}")
    print(f"{'='*60}\n")
    
    # Initialize components
    fetcher = AQICNFetcher()
    engineer = FeatureEngineer()
    db_handler = MongoDBHandler()
    
    try:
        #Fetch current data from AQICN
        print("Step 1: Fetching current AQI data...")
        raw_data = fetcher.fetch_current_data()
        
        if not raw_data:
            print("✗ Failed to fetch data from API")
            return False
        
        #Create features from raw data
        print("\nStep 2: Engineering features...")
        features = engineer.create_features(raw_data)
        
        if not features:
            print("✗ Failed to create features")
            return False
        
        #Store features in MongoDB
        print("\nStep 3: Storing features in MongoDB...")
        success = db_handler.insert_features(features)
        
        if not success:
            print("✗ Failed to store features")
            return False
        
        #Show statistics
        print("\nStep 4: Database statistics...")
        stats = db_handler.get_data_statistics()
        print(f"Total records in database: {stats['total_records']}")
        print(f"Date range: {stats['date_range_days']} days")
        
        #Check for hazardous AQI and send alerts
        print("\nStep 5: Monitoring AQI for alerts...")
        from src.alert_system import AlertSystem
        alert_system = AlertSystem()
        alert_system.monitor_aqi(features['aqi'])
        
        print(f"\n{'='*60}")
        print(f"✓ Feature Pipeline Completed Successfully!")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        return False
        
    finally:
        # Always close database connection
        db_handler.close()


def backfill_historical_data(days: int = 7):
    """
    Backfill historical data by running pipeline multiple times
    Note: This simulates historical data since free API doesn't provide it
    
    Args:
        days: Number of days to backfill (just runs pipeline multiple times)
    """
    print(f"\n{'='*60}")
    print(f"Backfill Process Started: {days} days")
    print(f"{'='*60}\n")
    print("Note: Free AQICN API doesn't provide historical data")
    print("We'll collect current data points to build history over time")
    print(f"Please run this pipeline hourly to collect sufficient data\n")
    
    success = run_feature_pipeline()
    
    if success:
        print("\n✓ First data point collected!")
        print("💡 Set up GitHub Actions to run this hourly for historical data")
    
    return success


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'backfill':
        # Run backfill mode
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        backfill_historical_data(days)
    else:
        # Run normal pipeline
        run_feature_pipeline()