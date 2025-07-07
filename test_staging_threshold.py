#!/usr/bin/env python3
"""
Test script to verify staging threshold and collection creation.
Creates enough similar documents to trigger the new collection logic.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import List

# Test documents - similar content to trigger clustering
SIMILAR_DOCUMENTS = [
    {
        "filename": "tech_article_1.txt",
        "content": "Artificial intelligence and machine learning are revolutionizing software development. Python frameworks like TensorFlow and PyTorch enable developers to build sophisticated AI applications. Neural networks process data patterns to make predictions and automate decision-making processes."
    },
    {
        "filename": "tech_article_2.txt", 
        "content": "Machine learning algorithms and artificial intelligence systems are transforming how we develop software. Libraries such as TensorFlow and PyTorch provide powerful tools for creating neural networks that can analyze data patterns and automate complex tasks."
    },
    {
        "filename": "tech_article_3.txt",
        "content": "The field of artificial intelligence continues to advance with machine learning techniques. Python developers use frameworks like TensorFlow and PyTorch to implement neural networks that recognize patterns in data and enable automated decision making."
    },
    {
        "filename": "tech_article_4.txt",
        "content": "Software development is being transformed by artificial intelligence and machine learning technologies. Neural networks built with TensorFlow and PyTorch can process large datasets to identify patterns and make intelligent predictions."
    },
    {
        "filename": "tech_article_5.txt",
        "content": "Modern AI applications rely on machine learning algorithms to process information. Python frameworks like TensorFlow and PyTorch allow developers to create neural networks that learn from data patterns and automate complex workflows."
    },
    {
        "filename": "tech_article_6.txt",
        "content": "Artificial intelligence powered by machine learning is reshaping software engineering. Tools like TensorFlow and PyTorch enable the creation of neural networks that can analyze data, recognize patterns, and make automated decisions."
    },
    {
        "filename": "tech_article_7.txt",
        "content": "The integration of artificial intelligence and machine learning in software development has accelerated innovation. Python libraries such as TensorFlow and PyTorch provide the foundation for building neural networks that process data and extract meaningful patterns."
    },
    {
        "filename": "tech_article_8.txt",
        "content": "Machine learning and AI technologies are becoming essential in modern software development. Frameworks like TensorFlow and PyTorch allow developers to implement neural networks that can learn from data patterns and automate intelligent processes."
    },
    {
        "filename": "tech_article_9.txt",
        "content": "Artificial intelligence systems leverage machine learning to solve complex problems in software development. Neural networks created with TensorFlow and PyTorch can analyze vast amounts of data to identify patterns and make predictive decisions."
    },
    {
        "filename": "tech_article_10.txt",
        "content": "The evolution of artificial intelligence and machine learning has transformed software engineering practices. Python developers use TensorFlow and PyTorch to build neural networks that process data patterns and enable sophisticated automation."
    },
    {
        "filename": "tech_article_11.txt",
        "content": "Machine learning algorithms powered by artificial intelligence are revolutionizing how we approach software development. Libraries like TensorFlow and PyTorch provide the tools needed to create neural networks that learn from data and make intelligent predictions."
    },
    {
        "filename": "tech_article_12.txt",
        "content": "The advancement of artificial intelligence through machine learning has opened new possibilities in software development. Neural networks built using TensorFlow and PyTorch can process complex data patterns and automate decision-making workflows."
    }
]

BASE_URL = "http://localhost:8000"

async def upload_document(session: aiohttp.ClientSession, doc: dict) -> dict:
    """Upload a single document and return the response."""
    
    # Create form data
    data = aiohttp.FormData()
    data.add_field('file', 
                   doc['content'].encode('utf-8'),
                   filename=doc['filename'],
                   content_type='text/plain')
    
    try:
        async with session.post(f"{BASE_URL}/process", data=data) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "filename": doc['filename'],
                    "success": True,
                    "response": result
                }
            else:
                error_text = await response.text()
                return {
                    "filename": doc['filename'],
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}"
                }
    except Exception as e:
        return {
            "filename": doc['filename'],
            "success": False,
            "error": str(e)
        }

async def check_ecosystem_status(session: aiohttp.ClientSession) -> dict:
    """Check the current ecosystem status."""
    try:
        async with session.get(f"{BASE_URL}/ecosystem") as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def list_collections(session: aiohttp.ClientSession) -> dict:
    """List all collections."""
    try:
        async with session.get(f"{BASE_URL}/collections") as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def main():
    """Main test function."""
    print("🧪 Testing Staging Threshold and Collection Creation")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Check initial state
        print("\n📊 Initial ecosystem status:")
        initial_status = await check_ecosystem_status(session)
        if "error" not in initial_status:
            ecosystem = initial_status.get("ecosystem", {})
            print(f"  Staging documents: {ecosystem.get('ecosystem_status', {}).get('documents_in_staging', 0)}")
            print(f"  Mature collections: {ecosystem.get('ecosystem_status', {}).get('mature_collections', 0)}")
            print(f"  Collection seeds: {ecosystem.get('ecosystem_status', {}).get('collection_seeds', 0)}")
        else:
            print(f"  Error: {initial_status['error']}")
        
        # Upload documents one by one and track progress
        print(f"\n📤 Uploading {len(SIMILAR_DOCUMENTS)} similar documents...")
        print("   (Should trigger collection creation after staging threshold)")
        
        results = []
        for i, doc in enumerate(SIMILAR_DOCUMENTS, 1):
            print(f"\n  📄 Uploading {i}/{len(SIMILAR_DOCUMENTS)}: {doc['filename']}")
            
            result = await upload_document(session, doc)
            results.append(result)
            
            if result["success"]:
                response = result["response"]
                print(f"    ✅ Success - Document ID: {response.get('organic_doc_id', 'N/A')}")
                
                # Check if we're still in staging
                processing_result = response.get('processing_result', {})
                if processing_result.get('staged'):
                    print(f"    📦 Document staged (waiting for threshold)")
                
                # Check ecosystem status after key uploads
                if i in [5, 10, 12]:  # Check at 5, 10, and final upload
                    print(f"    📊 Ecosystem status after {i} uploads:")
                    status = await check_ecosystem_status(session)
                    if "error" not in status:
                        ecosystem = status.get("ecosystem", {})
                        staging_count = ecosystem.get('ecosystem_status', {}).get('documents_in_staging', 0)
                        mature_count = ecosystem.get('ecosystem_status', {}).get('mature_collections', 0)
                        seed_count = ecosystem.get('ecosystem_status', {}).get('collection_seeds', 0)
                        
                        print(f"      - Staging: {staging_count} documents")
                        print(f"      - Mature collections: {mature_count}")
                        print(f"      - Seeds: {seed_count}")
                        
                        if staging_count == 0 and mature_count > 0:
                            print(f"    🎉 Collection created! Documents moved from staging.")
                            break
                    else:
                        print(f"    ❌ Status check failed: {status['error']}")
                        
            else:
                print(f"    ❌ Failed: {result['error']}")
            
            # Small delay between uploads
            await asyncio.sleep(0.5)
        
        # Final ecosystem status
        print(f"\n📊 Final ecosystem status:")
        final_status = await check_ecosystem_status(session)
        if "error" not in final_status:
            ecosystem = final_status.get("ecosystem", {})
            print(f"  Staging documents: {ecosystem.get('ecosystem_status', {}).get('documents_in_staging', 0)}")
            print(f"  Mature collections: {ecosystem.get('ecosystem_status', {}).get('mature_collections', 0)}")
            print(f"  Collection seeds: {ecosystem.get('ecosystem_status', {}).get('collection_seeds', 0)}")
            print(f"  Total organized: {ecosystem.get('ecosystem_status', {}).get('total_organized_documents', 0)}")
            print(f"  Health: {ecosystem.get('health_summary', {}).get('health_percentage', 0)}%")
        else:
            print(f"  Error: {final_status['error']}")
        
        # List all collections
        print(f"\n📁 Collections created:")
        collections = await list_collections(session)
        if "error" not in collections:
            for collection in collections.get("collections", []):
                print(f"  - {collection['name']}: {collection['document_count']} documents ({collection['type']})")
        else:
            print(f"  Error: {collections['error']}")
        
        # Summary
        print(f"\n📈 Upload Summary:")
        successful = len([r for r in results if r["success"]])
        failed = len([r for r in results if not r["success"]])
        print(f"  Successful uploads: {successful}")
        print(f"  Failed uploads: {failed}")
        
        if failed > 0:
            print(f"\n❌ Failed uploads:")
            for result in results:
                if not result["success"]:
                    print(f"  - {result['filename']}: {result['error']}")

if __name__ == "__main__":
    print("🚀 Starting staging threshold test...")
    print("Make sure the application is running on http://localhost:8000")
    print()
    
    try:
        asyncio.run(main())
        print("\n✅ Test completed!")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")