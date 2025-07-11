#!/usr/bin/env python3
"""
Manually trigger collection processing for staged MLB documents
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pymongo
from core.organic_collections import OrganicCollectionManager

async def trigger_mlb_processing():
    """Manually trigger processing of staged MLB documents."""
    
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['intelligent_data']
    staging_collection = db['document_staging']
    
    print("🚀 Manually triggering MLB collection processing...")
    
    # Get one MLB document to use as a trigger
    mlb_doc = staging_collection.find_one({
        "source_file": {"$regex": r"schedule\.pdf$"},
        "collection_assigned": None
    })
    
    if not mlb_doc:
        print("❌ No unprocessed MLB documents found")
        return
    
    print(f"📄 Using '{mlb_doc['source_file']}' as trigger document")
    
    # Initialize organic manager
    organic_manager = OrganicCollectionManager(client)
    
    # Manually trigger collection opportunities evaluation
    try:
        print("⚙️ Triggering collection opportunity evaluation...")
        
        # This should process all staged documents and create collections
        await organic_manager._evaluate_collection_opportunities(
            str(mlb_doc['_id']), 
            mlb_doc
        )
        
        print("✅ Collection processing triggered successfully!")
        
        # Check results
        print("\n📊 Checking results...")
        
        # Count remaining staged MLB docs
        remaining_mlb = staging_collection.count_documents({
            "source_file": {"$regex": r"schedule\.pdf$"},
            "collection_assigned": None
        })
        
        print(f"📈 MLB documents still in staging: {remaining_mlb}")
        
        # Check for new collections
        registry = db['collection_registry']
        recent_collections = list(registry.find({}).sort([("created_at", -1)]).limit(3))
        
        print(f"\n🆕 Recent collections:")
        for collection in recent_collections:
            name = collection['collection_name']
            count = collection['document_count']
            created = collection['created_at'].strftime("%H:%M:%S")
            keywords = collection.get('theme_keywords', [])[:3]
            print(f"   {name}: {count} docs at {created}, keywords: {keywords}")
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(trigger_mlb_processing())