#!/usr/bin/env python3
"""
Test script for business document classifier
"""

import asyncio
from pathlib import Path
from core.business_classifier import BusinessDocumentClassifier
from core.business_document_types import BusinessDocumentType

async def test_business_classifier():
    classifier = BusinessDocumentClassifier()
    
    # Test with our sample documents
    sample_docs = Path("sample_business_docs")
    
    print("🔍 Testing Business Document Classifier")
    print("=" * 50)
    
    for doc_file in sample_docs.glob("*.txt"):
        try:
            content = doc_file.read_text()
            result = await classifier.classify_business_document(content, doc_file.name)
            
            print(f"\n📄 File: {doc_file.name}")
            print(f"   Type: {result.business_type.value if result.business_type else 'None'}")
            print(f"   Category: {result.category}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Collection: {result.suggested_collection}")
            print(f"   Reasoning: {result.reasoning[:100]}...")
            
            if result.metadata:
                print(f"   Metadata: {list(result.metadata.keys())}")
        
        except Exception as e:
            print(f"❌ Error processing {doc_file.name}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Business classifier test completed")

if __name__ == "__main__":
    asyncio.run(test_business_classifier())