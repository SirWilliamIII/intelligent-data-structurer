#!/usr/bin/env python3
"""
Real-time monitoring of NFL upload process
"""

import asyncio
import time
import pymongo
from datetime import datetime

async def monitor_nfl_process():
    """Monitor the NFL upload process in real-time."""
    
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['intelligent_data']
    
    print("🏈 NFL Upload Monitoring Started")
    print("=" * 50)
    
    last_staging_count = 0
    last_collection_count = 0
    monitoring = True
    start_time = time.time()
    
    while monitoring:
        try:
            # Check staging pool
            staging_count = 0
            collections_info = {}
            
            if 'document_staging' in db.list_collection_names():
                staging_count = db.document_staging.count_documents({})
                
                # Count NFL schedules specifically
                nfl_count = db.document_staging.count_documents({
                    "source_file": {"$regex": r".*schedule\.pdf$"}
                })
                
                if nfl_count > 0:
                    print(f"📄 NFL schedules in staging: {nfl_count}")
                    
                    # Sample semantic signature
                    sample = db.document_staging.find_one({
                        "source_file": {"$regex": r".*schedule\.pdf$"}
                    }, {"semantic_signature": 1, "source_file": 1})
                    
                    if sample and sample.get('semantic_signature'):
                        keywords = sample['semantic_signature'].get('domain_keywords', [])[:5]
                        print(f"   Sample keywords: {keywords}")
            
            # Check collections
            collection_names = db.list_collection_names()
            total_collections = len(collection_names)
            
            for col_name in collection_names:
                if col_name not in ['collection_registry', 'collection_seeds']:
                    count = db[col_name].count_documents({})
                    collections_info[col_name] = count
            
            # Check collection registry
            if 'collection_registry' in collection_names:
                registry = list(db.collection_registry.find().sort([("created_at", -1)]))
                if registry:
                    latest = registry[0]
                    print(f"🆕 Latest collection: {latest['collection_name']} ({latest['document_count']} docs)")
                    print(f"   Keywords: {latest.get('theme_keywords', [])}")
                    print(f"   Created: {latest['created_at'].strftime('%H:%M:%S')}")
            
            # Check seeds
            if 'collection_seeds' in collection_names:
                seeds = list(db.collection_seeds.find())
                if seeds:
                    print(f"🌱 Active seeds: {len(seeds)}")
                    for seed in seeds:
                        print(f"   Theme: {seed.get('theme_keywords', [])} ({len(seed.get('document_ids', []))} docs)")
            
            # Print summary
            elapsed = int(time.time() - start_time)
            print(f"\n⏱️  [{elapsed}s] Staging: {staging_count} | Collections: {total_collections} | Active: {collections_info}")
            
            # Detect changes
            if staging_count != last_staging_count or total_collections != last_collection_count:
                print("🔄 Change detected!")
                last_staging_count = staging_count
                last_collection_count = total_collections
            
            print("-" * 50)
            
            # Stop monitoring if process seems complete
            if staging_count == 0 and total_collections > 0:
                print("✅ Upload process appears complete!")
                monitoring = False
            
            await asyncio.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            monitoring = False
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            await asyncio.sleep(5)
    
    print("\n📊 Final State:")
    collection_names = db.list_collection_names()
    for col_name in collection_names:
        count = db[col_name].count_documents({})
        print(f"   {col_name}: {count} documents")

if __name__ == "__main__":
    asyncio.run(monitor_nfl_process())