"""
Remove duplicate records for the same hour, keeping only one per hour
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mongodb_handler import MongoDBHandler
from datetime import datetime
from collections import defaultdict

def cleanup_duplicates():
    db = MongoDBHandler()
    
    # Get all records
    records = list(db.db.features.find({}).sort('timestamp', 1))
    print(f"Total records: {len(records)}")
    
    # Group by hour
    by_hour = defaultdict(list)
    for r in records:
        hour = r['timestamp'].replace(minute=0, second=0, microsecond=0)
        by_hour[hour].append(r)
    
    # Find and remove duplicates
    total_deleted = 0
    for hour, recs in sorted(by_hour.items()):
        if len(recs) > 1:
            print(f"\nHour {hour}: {len(recs)} records")
            for r in recs:
                print(f"  - {r['timestamp']} AQI={r.get('aqi')} (inserted: {r.get('inserted_at')})")
            
            # Keep the most recently inserted one (likely real-time data)
            recs.sort(key=lambda x: x.get('inserted_at', datetime.min), reverse=True)
            to_delete = [r['_id'] for r in recs[1:]]
            
            if to_delete:
                db.db.features.delete_many({'_id': {'$in': to_delete}})
                kept = recs[0]
                print(f"  -> Kept: {kept['timestamp']} AQI={kept.get('aqi')}")
                print(f"  -> Deleted: {len(to_delete)} duplicates")
                total_deleted += len(to_delete)
    
    print(f"\n{'='*50}")
    print(f"Total duplicates removed: {total_deleted}")
    print(f"Records remaining: {db.db.features.count_documents({})}")
    db.close()

if __name__ == "__main__":
    cleanup_duplicates()
