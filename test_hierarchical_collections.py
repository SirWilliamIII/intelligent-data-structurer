#!/usr/bin/env python3
"""
Test the hierarchical collection subdivision system
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pymongo
from core.organic_collections import OrganicCollectionManager

async def test_hierarchical_subdivision():
    """Test if the system can create hierarchical subdivisions."""
    
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017')
    organic_manager = OrganicCollectionManager(client)
    
    print("🌳 Testing Hierarchical Collection Subdivision")
    print("=" * 50)
    
    # Check current collection state
    print("\n📊 Current Collection State:")
    db = client['intelligent_data']
    collection_registry = db['collection_registry']
    
    collections = list(collection_registry.find({"status": "mature"}))
    for collection in collections:
        name = collection['collection_name']
        count = collection['document_count']
        keywords = collection.get('theme_keywords', [])
        print(f"   {name}: {count} docs, keywords: {keywords}")
    
    # Test subdivision monitoring on event_info_documents (which has NFL schedules)
    print(f"\n🔍 Testing subdivision of 'event_info_documents'...")
    
    try:
        # Manually trigger subdivision analysis
        await organic_manager._monitor_collection_subdivision()
        
        print("\n📈 Post-Subdivision Collection State:")
        
        # Check if new subdivisions were created
        updated_collections = list(collection_registry.find({"status": "mature"}))
        
        new_collections = []
        for collection in updated_collections:
            if collection not in collections:  # New collection
                new_collections.append(collection)
        
        if new_collections:
            print("   ✅ New subdivisions created:")
            for collection in new_collections:
                name = collection['collection_name']
                count = collection['document_count']
                parent = collection.get('parent_collection', 'None')
                keywords = collection.get('theme_keywords', [])
                print(f"      {name}: {count} docs from '{parent}', keywords: {keywords}")
        else:
            print("   ⏳ No subdivisions created yet (may need more distinct patterns)")
            
            # Check what patterns were found
            event_collection = db['event_info_documents']
            sample_docs = list(event_collection.find({}, {"semantic_signature": 1}).limit(5))
            
            print(f"\n📋 Sample keywords from event_info_documents:")
            for i, doc in enumerate(sample_docs):
                sig = doc.get('semantic_signature', {})
                keywords = sig.get('domain_keywords', [])[:5]
                print(f"      Doc {i+1}: {keywords}")
                
    except Exception as e:
        print(f"❌ Error during subdivision test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hierarchical_subdivision())