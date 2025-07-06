#!/usr/bin/env python3
"""
Simple script to check MongoDB entries.
"""

import asyncio
from core.mongo_database import mongo_manager
from pprint import pprint

async def check_mongodb():
    """Check MongoDB collections and their entries."""
    try:
        # Connect to MongoDB
        await mongo_manager.connect()
        
        # Get collection statistics
        print("MongoDB Collection Statistics:")
        print("=" * 50)
        
        stats = await mongo_manager.get_collection_stats()
        for collection_name, count in stats.items():
            print(f"{collection_name}: {count} documents")
        
        print("\n" + "=" * 50)
        
        # Show sample documents from each collection
        collections = await mongo_manager.list_collections()
        for collection_name in collections:
            print(f"\n--- Sample documents from '{collection_name}' ---")
            documents = await mongo_manager.find_documents(collection_name, limit=5)
            
            if documents:
                for i, doc in enumerate(documents, 1):
                    print(f"\nDocument {i}:")
                    # Show key fields for better readability
                    for key in ['_id', 'content_type', 'domain', 'priority', 'source_file', 'extracted_at', 'inserted_at']:
                        if key in doc:
                            print(f"  {key}: {doc[key]}")
                    
                    # Show content preview if available
                    if 'content' in doc:
                        content = doc['content']
                        if isinstance(content, str) and len(content) > 200:
                            print(f"  content: {content[:200]}...")
                        else:
                            print(f"  content: {content}")
                    
                    print("-" * 30)
            else:
                print("  No documents found")
        
    except Exception as e:
        print(f"Error checking MongoDB: {e}")
    finally:
        await mongo_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(check_mongodb())