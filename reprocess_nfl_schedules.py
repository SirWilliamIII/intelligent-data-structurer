#!/usr/bin/env python3
"""
Reprocess the NFL schedules in staging with improved semantic enrichment
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

import pymongo
from core.intelligent_analyzer import IntelligentAnalyzer
from core.organic_collections import OrganicCollectionManager

async def reprocess_nfl_schedules():
    """Reprocess NFL schedules with new semantic enrichment."""
    
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['intelligent_data']
    staging_collection = db['document_staging']
    
    # Get NFL schedule documents
    nfl_docs = list(staging_collection.find({"source_file": {"$regex": r"schedule\.pdf$"}}))
    
    print(f"🏈 Found {len(nfl_docs)} NFL schedule documents in staging")
    
    if not nfl_docs:
        print("❌ No NFL schedules found in staging")
        return
    
    # Initialize components
    analyzer = IntelligentAnalyzer()
    organic_manager = OrganicCollectionManager(client)
    
    # Process first few documents to test
    test_docs = nfl_docs[:3]
    
    for i, doc in enumerate(test_docs):
        filename = doc['source_file']
        content = doc['content']
        
        print(f"\n📄 Processing {i+1}/{len(test_docs)}: {filename}")
        
        try:
            # Reanalyze with improved semantic enrichment
            result = await analyzer.analyze_content(content, filename)
            
            print(f"   Collection: {result.table_name}")
            print(f"   Confidence: {result.confidence:.2f}")
            
            # Check keywords
            keywords = result.extracted_data.get('semantic_signature', {}).get('domain_keywords', [])
            sports_keywords = [kw for kw in keywords if kw in ['nfl', 'football', 'team', 'sport', 'league']]
            
            if sports_keywords:
                print(f"   ✅ Sports keywords: {sports_keywords}")
            else:
                print(f"   ❌ No sports keywords found")
                print(f"   Top keywords: {keywords[:3]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(reprocess_nfl_schedules())