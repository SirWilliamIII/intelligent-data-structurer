#!/usr/bin/env python3
"""
Demo queries showcasing the power of AI-organized data
"""

import pymongo
from datetime import datetime
import json
from collections import defaultdict

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['intelligent_data']

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def run_power_queries():
    """Run queries that showcase the intelligence of your data organization"""
    
    print_separator("1. CROSS-COLLECTION INTELLIGENCE: Find all documents with contact info")
    
    # Find all documents across collections that contain emails or phones
    contact_docs = []
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        
        # Query for documents with emails or phones
        docs_with_contacts = list(collection.find({
            "$or": [
                {"emails": {"$ne": ""}},
                {"phones": {"$ne": ""}},
                {"emails": {"$exists": True, "$ne": []}},
                {"phones": {"$exists": True, "$ne": []}}
            ]
        }, {"source_file": 1, "emails": 1, "phones": 1, "content": 1}))
        
        for doc in docs_with_contacts:
            contact_docs.append({
                "collection": collection_name,
                "file": doc.get("source_file", "Unknown"),
                "emails": doc.get("emails", ""),
                "phones": doc.get("phones", ""),
                "preview": doc.get("content", "")[:100] + "..."
            })
    
    print(f"Found {len(contact_docs)} documents with contact information across collections:")
    for doc in contact_docs[:5]:  # Show first 5
        print(f"  📁 {doc['collection']}: {doc['file']}")
        if doc['emails']:
            print(f"    📧 Emails: {doc['emails']}")
        if doc['phones']:
            print(f"    📞 Phones: {doc['phones']}")
        print()

    print_separator("2. ENTITY INTELLIGENCE: Find documents mentioning specific organizations")
    
    # Search for documents mentioning specific entities across all collections
    target_entities = ["Google", "Microsoft", "Apple", "Amazon", "Meta"]
    
    for entity in target_entities:
        matching_docs = []
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            
            # Search in content and entities fields
            docs = list(collection.find({
                "$or": [
                    {"content": {"$regex": entity, "$options": "i"}},
                    {"entities": {"$regex": entity, "$options": "i"}}
                ]
            }, {"source_file": 1, "content": 1, "classification_confidence": 1}))
            
            for doc in docs:
                matching_docs.append({
                    "collection": collection_name,
                    "file": doc.get("source_file", "Unknown"),
                    "confidence": doc.get("classification_confidence", "N/A"),
                    "preview": doc.get("content", "")[:150] + "..."
                })
        
        if matching_docs:
            print(f"📊 {entity} mentioned in {len(matching_docs)} documents:")
            for doc in matching_docs[:3]:  # Show first 3
                print(f"  📁 {doc['collection']}: {doc['file']} (confidence: {doc['confidence']})")
                print(f"    Preview: {doc['preview']}")
            print()

    print_separator("3. CLASSIFICATION CONFIDENCE ANALYSIS")
    
    # Analyze classification confidence across collections
    confidence_stats = defaultdict(list)
    
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        
        docs = list(collection.find({
            "classification_confidence": {"$exists": True}
        }, {"classification_confidence": 1, "classification_reasoning": 1, "source_file": 1}))
        
        for doc in docs:
            confidence_stats[collection_name].append({
                "confidence": doc.get("classification_confidence", 0),
                "reasoning": doc.get("classification_reasoning", ""),
                "file": doc.get("source_file", "Unknown")
            })
    
    print("📊 Classification Quality by Collection:")
    for collection, stats in confidence_stats.items():
        if stats:
            avg_confidence = sum(s["confidence"] for s in stats) / len(stats)
            print(f"  📁 {collection}: {avg_confidence:.1%} avg confidence ({len(stats)} docs)")
            
            # Show best classified document
            best_doc = max(stats, key=lambda x: x["confidence"])
            print(f"    🏆 Best: {best_doc['file']} ({best_doc['confidence']:.1%})")
            print(f"    💡 Reasoning: {best_doc['reasoning'][:100]}...")
            print()

    print_separator("4. CONTENT TYPE DISTRIBUTION")
    
    # Analyze what types of content were found
    content_types = defaultdict(int)
    
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        
        docs = list(collection.find({
            "content_type": {"$exists": True}
        }, {"content_type": 1}))
        
        for doc in docs:
            content_type = doc.get("content_type", "Unknown")
            content_types[content_type] += 1
    
    print("📊 Content Types Identified:")
    for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  📄 {content_type}: {count} documents")

    print_separator("5. SEMANTIC SIMILARITY SEARCH")
    
    # Find documents with similar semantic signatures
    print("🔍 Finding documents with similar content patterns...")
    
    # Get a sample document's semantic signature
    sample_collection = db['technical_documents']
    sample_doc = sample_collection.find_one({"semantic_signature": {"$exists": True}})
    
    if sample_doc and sample_doc.get("semantic_signature"):
        print(f"📋 Using semantic signature from: {sample_doc.get('source_file', 'Unknown')}")
        
        # Find similar documents across all collections
        similar_docs = []
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            
            # This is a simplified similarity - in practice you'd use vector similarity
            docs = list(collection.find({
                "semantic_signature": {"$exists": True}
            }, {"source_file": 1, "semantic_signature": 1, "content": 1}))
            
            for doc in docs:
                if doc["_id"] != sample_doc["_id"]:  # Don't include the same document
                    similar_docs.append({
                        "collection": collection_name,
                        "file": doc.get("source_file", "Unknown"),
                        "preview": doc.get("content", "")[:100] + "..."
                    })
        
        print(f"📊 Found {len(similar_docs)} potentially similar documents:")
        for doc in similar_docs[:5]:
            print(f"  📁 {doc['collection']}: {doc['file']}")
            print(f"    Preview: {doc['preview']}")
            print()

    print_separator("6. BUSINESS INTELLIGENCE: Document Processing Stats")
    
    # Overall statistics
    total_docs = 0
    total_collections = len(db.list_collection_names())
    word_counts = []
    file_types = defaultdict(int)
    
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        count = collection.count_documents({})
        total_docs += count
        
        # Get word count statistics
        docs = list(collection.find({
            "word_count": {"$exists": True}
        }, {"word_count": 1, "source_file": 1}))
        
        for doc in docs:
            word_counts.append(doc.get("word_count", 0))
            
            # Extract file extension
            filename = doc.get("source_file", "")
            if "." in filename:
                ext = filename.split(".")[-1].lower()
                file_types[ext] += 1
    
    print(f"📊 Processing Summary:")
    print(f"  📁 Total Collections: {total_collections}")
    print(f"  📄 Total Documents: {total_docs}")
    
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        print(f"  📝 Average Words per Document: {avg_words:.0f}")
        print(f"  📈 Word Count Range: {min(word_counts)} - {max(word_counts)}")
    
    print(f"  📋 File Types Processed:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    .{ext}: {count} files")

if __name__ == "__main__":
    print("🚀 INTELLIGENT DATA PROCESSOR - POWER QUERY DEMO")
    print("Demonstrating AI-powered data organization and retrieval")
    
    try:
        run_power_queries()
        
        print(f"\n{'='*60}")
        print("✅ DEMO COMPLETE!")
        print("💡 This demonstrates how your AI automatically:")
        print("   • Classifies documents by content type")
        print("   • Extracts structured data (emails, phones, entities)")
        print("   • Provides confidence scoring for quality assurance")
        print("   • Enables cross-collection intelligent search")
        print("   • Creates semantic fingerprints for similarity matching")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Make sure MongoDB is running and contains data")