"""
Cleanup script to remove future/forecast data that was accidentally stored
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mongodb_handler import MongoDBHandler
from datetime import datetime

def cleanup_future_records():
    """Remove any records with timestamps in the future"""
    db = MongoDBHandler()
    
    now = datetime.utcnow()
    print(f"Current UTC time: {now}")
    
    # Find future records
    future_records = list(db.db.features.find({
        'timestamp': {'$gt': now}
    }))
    
    print(f"\nFound {len(future_records)} records with future timestamps:")
    for r in future_records[:10]:  # Show first 10
        print(f"  - {r['timestamp']} (inserted: {r.get('inserted_at', 'N/A')})")
    
    # Also check for records where timestamp > inserted_at by more than 1 day
    # This catches cases where forecast data was stored
    suspicious = list(db.db.features.find({}).limit(1000))
    bad_records = []
    for r in suspicious:
        ts = r.get('timestamp')
        inserted = r.get('inserted_at')
        if ts and inserted and ts > inserted:
            diff = (ts - inserted).days
            if diff > 0:  # Timestamp is future relative to when it was inserted
                bad_records.append(r)
    
    print(f"\nFound {len(bad_records)} records where timestamp > inserted_at (forecast data):")
    for r in bad_records[:10]:
        print(f"  - timestamp: {r['timestamp']}, inserted_at: {r.get('inserted_at')}")
    
    if future_records or bad_records:
        # Get all IDs to delete
        ids_to_delete = set()
        for r in future_records:
            ids_to_delete.add(r['_id'])
        for r in bad_records:
            ids_to_delete.add(r['_id'])
        
        if ids_to_delete:
            confirm = input(f"\nDelete {len(ids_to_delete)} bad records? (y/n): ")
            if confirm.lower() == 'y':
                result = db.db.features.delete_many({'_id': {'$in': list(ids_to_delete)}})
                print(f"Deleted {result.deleted_count} records")
            else:
                print("Skipped deletion")
    else:
        print("\nNo bad records found!")
    
    db.close()

if __name__ == "__main__":
    cleanup_future_records()
