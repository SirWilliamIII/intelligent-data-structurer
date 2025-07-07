#!/usr/bin/env python3
"""
Simple test of the organic collections system
"""

import asyncio
import pymongo
from core.organic_collections import OrganicCollectionManager
from rich.console import Console

console = Console()

async def test_organic_system():
    """Test the organic system with minimal demo."""
    
    console.print("🧠 Testing Organic Collection System...")
    
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    
    # Initialize organic manager
    organic_manager = OrganicCollectionManager(client)
    
    # Get current ecosystem status
    insights = await organic_manager.get_collection_insights()
    
    console.print(f"📊 Current Ecosystem:")
    console.print(f"   • Documents in staging: {insights['ecosystem_status']['documents_in_staging']}")
    console.print(f"   • Collection seeds: {insights['ecosystem_status']['collection_seeds']}")
    console.print(f"   • Mature collections: {insights['ecosystem_status']['mature_collections']}")
    console.print(f"   • Organized documents: {insights['ecosystem_status']['total_organized_documents']}")
    
    # Test with a simple document
    test_doc = {
        "content": "# Kubernetes Tutorial\n\nThis is a simple guide for using kubectl commands.\n\n```bash\nkubectl get pods\nkubectl apply -f deployment.yaml\n```",
        "source_file": "kubernetes-test.md",
        "semantic_signature": {
            "domain_keywords": ["kubernetes", "kubectl", "tutorial", "commands"],
            "structural_patterns": ["code_blocks", "technical_headers"],
            "entity_types": ["PRODUCT"],
            "content_markers": ["technical_content"],
            "semantic_hash": "test123"
        },
        "word_count": 15,
        "char_count": 150
    }
    
    console.print("\n🔄 Processing test document...")
    doc_id = await organic_manager.process_document(test_doc)
    console.print(f"✅ Document processed: {doc_id}")
    
    # Check updated status
    insights = await organic_manager.get_collection_insights()
    console.print(f"\n📊 Updated Ecosystem:")
    console.print(f"   • Documents in staging: {insights['ecosystem_status']['documents_in_staging']}")
    console.print(f"   • Collection seeds: {insights['ecosystem_status']['collection_seeds']}")
    
    console.print("\n✅ Basic organic system test complete!")

if __name__ == "__main__":
    asyncio.run(test_organic_system())