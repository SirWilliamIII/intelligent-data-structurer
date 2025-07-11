#!/usr/bin/env python3
"""
Test Intelligent Taxonomy System

Test the new intelligent system with controlled document reprocessing.
Validates that the system creates semantically meaningful collections
instead of overly specific ones.
"""

import asyncio
import sys
import os
import pymongo
import json
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(__file__))

from core.intelligent_collection_engine import IntelligentCollectionEngine
from loguru import logger

async def test_intelligent_system():
    """
    Test the intelligent taxonomy system with sample documents.
    
    This validates that we get intelligent collections like:
    - 'nfl_team_schedules' instead of 'chicago_bulls_game_64_documents'
    - Proper semantic hierarchy and organization
    """
    
    print("🧪 TESTING INTELLIGENT TAXONOMY SYSTEM")
    print("=" * 45)
    print()
    print("🎯 Goal: Prove the system creates intelligent collections")
    print("❌ No more: 'chicago_bulls_game_64_documents'") 
    print("✅ Instead: 'nfl_team_schedules', 'sports_contracts', etc.")
    print()
    
    # Connect to MongoDB
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017')
        db = client['intelligent_data']
        client.admin.command('ping')
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False
    
    # Initialize intelligent engine
    print("🧠 Initializing Intelligent Collection Engine...")
    engine = IntelligentCollectionEngine(client)
    
    # Check system health
    health = await engine.validate_system_health()
    if health["overall_status"] != "healthy":
        print(f"⚠️ System health check failed: {health['issues']}")
        return False
    
    print("✅ All systems operational")
    
    # Create test documents
    test_documents = create_test_documents()
    
    print(f"\n📄 Created {len(test_documents)} test documents:")
    for doc in test_documents:
        print(f"   • {doc['source_file']}")
    
    # Process documents through intelligent pipeline
    print(f"\n🚀 Processing documents through intelligent pipeline...")
    print("🧠 This will demonstrate:")
    print("   • Claude semantic analysis")
    print("   • Volume-based collection creation")
    print("   • Intelligent taxonomy organization")
    print()
    
    results = []
    for i, document in enumerate(test_documents):
        print(f"📄 Processing {i+1}/{len(test_documents)}: {document['source_file']}")
        
        # Add to staging first
        db.document_staging.insert_one(document)
        
        # Process through intelligent pipeline
        result = await engine.process_document(document)
        results.append(result)
        
        # Show result
        print(f"   → Collection: {result.collection_assigned}")
        print(f"   → Confidence: {result.confidence:.2f}")
        print(f"   → Reasoning: {result.reasoning[:100]}...")
        
        if result.created_collections:
            print(f"   🆕 Created new collections: {result.created_collections}")
        
        print()
        
        # Small delay to simulate real processing
        await asyncio.sleep(0.5)
    
    # Analyze results
    print("📊 ANALYSIS OF RESULTS")
    print("=" * 25)
    
    # Show collections created
    collections = db.list_collection_names()
    document_collections = [c for c in collections if c not in ['collection_registry', 'collection_seeds', 'document_staging']]
    
    print(f"\n🗂️ Collections Created: {len(document_collections)}")
    for collection_name in document_collections:
        count = db[collection_name].count_documents({})
        print(f"   • {collection_name}: {count} documents")
    
    # Show taxonomy structure
    print(f"\n🌳 Taxonomy Structure:")
    registry_docs = list(db.collection_registry.find({}))
    
    for registry_entry in registry_docs:
        name = registry_entry['collection_name']
        parent = registry_entry.get('hierarchy', {}).get('parent_collection', 'ROOT')
        level = registry_entry.get('hierarchy', {}).get('abstraction_level', 'unknown')
        concepts = registry_entry.get('ontology_profile', {}).get('primary_concepts', [])
        
        indent = "   " if parent != 'ROOT' else ""
        print(f"{indent}📁 {name} ({level})")
        print(f"{indent}   Parent: {parent}")
        print(f"{indent}   Concepts: {concepts}")
        print()
    
    # Validate intelligence  
    print("🧠 INTELLIGENCE VALIDATION")
    print("=" * 30)
    
    intelligence_score = 0
    max_score = 0
    
    # Test 1: No overly specific collections
    max_score += 20
    overly_specific = [c for c in document_collections if 'game_' in c and len(c.split('_')) > 4]
    if not overly_specific:
        intelligence_score += 20
        print("✅ No overly specific collections (like 'game_64_documents')")
    else:
        print(f"❌ Found overly specific collections: {overly_specific}")
    
    # Test 2: Semantic grouping
    max_score += 20
    semantic_collections = [c for c in document_collections if any(keyword in c for keyword in ['nfl', 'nba', 'sports', 'contracts', 'schedules'])]
    if len(semantic_collections) >= len(document_collections) * 0.8:
        intelligence_score += 20
        print("✅ Collections use semantic naming")
    else:
        print(f"❌ Collections not semantically named: {document_collections}")
    
    # Test 3: Appropriate abstraction levels
    max_score += 20
    medium_level_collections = [r for r in registry_docs if r.get('hierarchy', {}).get('abstraction_level') == 'medium']
    if len(medium_level_collections) > 0:
        intelligence_score += 20
        print("✅ Uses appropriate abstraction levels")
    else:
        print("❌ No medium-level abstractions found")
    
    # Test 4: Document type separation
    max_score += 20
    type_specific_collections = [c for c in document_collections if any(t in c for t in ['schedule', 'contract'])]
    if len(type_specific_collections) > 0:
        intelligence_score += 20
        print("✅ Separates different document types")
    else:
        print("❌ No document type separation")
    
    # Test 5: Confidence levels
    max_score += 20
    high_confidence_results = [r for r in results if r.confidence >= 0.7]
    if len(high_confidence_results) >= len(results) * 0.6:
        intelligence_score += 20
        print(f"✅ High confidence placements: {len(high_confidence_results)}/{len(results)}")
    else:
        print(f"❌ Low confidence placements: {len(high_confidence_results)}/{len(results)}")
    
    # Final score
    intelligence_percentage = (intelligence_score / max_score) * 100
    print(f"\n🎯 INTELLIGENCE SCORE: {intelligence_score}/{max_score} ({intelligence_percentage:.1f}%)")
    
    if intelligence_percentage >= 80:
        print("🌟 EXCELLENT! The system is working brilliantly!")
    elif intelligence_percentage >= 60:
        print("✅ GOOD! The system shows intelligent behavior!")
    elif intelligence_percentage >= 40:
        print("⚠️ OKAY! The system needs some tuning.")
    else:
        print("❌ POOR! The system needs significant improvement.")
    
    # Show processing statistics
    stats = await engine.get_processing_statistics()
    print(f"\n📈 PROCESSING STATISTICS:")
    print(f"   • Total documents: {stats['processing_stats']['total_documents']}")
    print(f"   • Successful placements: {stats['processing_stats']['successful_placements']}")
    print(f"   • New collections created: {stats['processing_stats']['new_collections_created']}")
    print(f"   • Average confidence: {stats['processing_stats']['average_confidence']:.2f}")
    print(f"   • Average processing time: {stats['processing_stats']['average_processing_time']:.2f}s")
    
    return intelligence_percentage >= 60

def create_test_documents():
    """Create test documents to validate intelligent processing."""
    
    return [
        # NFL Schedule (should create nfl_team_schedules, not game-specific)
        {
            "_id": "nfl_test_1",
            "source_file": "Chicago_Bears_schedule.pdf",
            "content": """Chicago Bears - 2025 NFL Schedule
Game 1: vs Green Bay Packers
Game 2: vs Detroit Lions  
Game 3: vs Minnesota Vikings
Game 4: vs Dallas Cowboys
Game 5: vs New York Giants
Game 6: vs Philadelphia Eagles
Game 7: vs Washington Commanders
Game 8: vs Atlanta Falcons""",
            "inserted_at": datetime.utcnow(),
            "collection_assigned": None
        },
        
        # NBA Schedule (should create nba_team_schedules or sports_schedules)
        {
            "_id": "nba_test_1", 
            "source_file": "Lakers_schedule.pdf",
            "content": """Los Angeles Lakers - 2025 NBA Schedule
Game 1: vs Boston Celtics
Game 2: vs Golden State Warriors
Game 3: vs Miami Heat
Game 4: vs Chicago Bulls
Game 5: vs Denver Nuggets
Game 6: vs Phoenix Suns
Game 7: vs Dallas Mavericks""",
            "inserted_at": datetime.utcnow(),
            "collection_assigned": None
        },
        
        # NFL Contract (should create sports_contracts, separate from schedules)
        {
            "_id": "nfl_contract_test_1",
            "source_file": "Bears_player_contracts.pdf", 
            "content": """Chicago Bears - 2025 Player Contracts
Name: Justin Fields
Position: Quarterback
Years: 4
Guaranteed: $18,871,000
Total Value: $32,500,000

Name: David Montgomery  
Position: Running Back
Years: 2
Guaranteed: $8,000,000
Total Value: $12,000,000""",
            "inserted_at": datetime.utcnow(),
            "collection_assigned": None
        },
        
        # Another NFL Schedule (should go to same collection as first)
        {
            "_id": "nfl_test_2",
            "source_file": "Packers_schedule.pdf",
            "content": """Green Bay Packers - 2025 NFL Schedule  
Game 1: vs Chicago Bears
Game 2: vs Minnesota Vikings
Game 3: vs Detroit Lions
Game 4: vs Tampa Bay Buccaneers
Game 5: vs New Orleans Saints
Game 6: vs Carolina Panthers""",
            "inserted_at": datetime.utcnow(),
            "collection_assigned": None
        },
        
        # Business Document (should create separate business collection)
        {
            "_id": "business_test_1",
            "source_file": "Q4_Financial_Report.pdf",
            "content": """Q4 2024 Financial Report
Revenue: $245.6 million
Net Income: $34.2 million  
Operating Expenses: $189.4 million
EBITDA: $56.8 million

Key Metrics:
- Customer Growth: 12%
- Market Share: 8.4%
- Operating Margin: 14%""",
            "inserted_at": datetime.utcnow(), 
            "collection_assigned": None
        }
    ]

if __name__ == "__main__":
    try:
        success = asyncio.run(test_intelligent_system())
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 INTELLIGENT SYSTEM TEST PASSED!")
            print("✨ Ready for production document processing!")
        else:
            print("⚠️ System needs improvement before production use")
            
        print("\n💡 Next Steps:")
        print("   • Upload real documents to test further")
        print("   • Monitor collection evolution over time")
        print("   • Tune confidence thresholds if needed")
        print("   • Watch concept drift detection in action")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()