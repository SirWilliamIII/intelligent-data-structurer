#!/usr/bin/env python3
"""
Test the organic system by uploading documents via API
"""

import asyncio
import aiohttp
import os
from pathlib import Path

async def test_organic_uploads():
    """Test uploading documents to the organic system."""
    
    print("🧪 Testing Organic System with Document Uploads")
    print("=" * 50)
    
    # Test documents content
    test_docs = [
        {
            "filename": "kubernetes-guide.md",
            "content": """# Kubernetes Tutorial

## Installation

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl describe pod my-pod
```

This guide covers Docker containers and microservices deployment.
Learn about namespaces, services, and ingress controllers.
"""
        },
        {
            "filename": "react-tutorial.md", 
            "content": """# React Tutorial

## Setup

```javascript
npm install react
npm start
```

Learn about components, hooks, and state management.
Build modern web applications with JSX and TypeScript.
"""
        },
        {
            "filename": "chocolate-chip-cookies.md",
            "content": """# Chocolate Chip Cookies

## Ingredients
- 2 cups flour
- 1 cup butter
- 1/2 cup sugar
- 2 cups chocolate chips

## Instructions
1. Preheat oven to 350°F
2. Mix ingredients
3. Bake for 12 minutes
"""
        },
        {
            "filename": "football-teams.txt",
            "content": """Football Teams

Dallas Cowboys
New York Giants  
Green Bay Packers
Chicago Bears
Pittsburgh Steelers
New England Patriots
"""
        },
        {
            "filename": "python-cheatsheet.md",
            "content": """# Python Cheat Sheet

## List Operations
```python
my_list = [1, 2, 3]
my_list.append(4)
```

## Dictionary Operations  
```python
my_dict = {'key': 'value'}
my_dict.update({'new_key': 'new_value'})
```

Essential Python commands for beginners.
"""
        }
    ]
    
    # Upload each document
    async with aiohttp.ClientSession() as session:
        
        for i, doc in enumerate(test_docs):
            print(f"\n📄 Uploading {doc['filename']} ({i+1}/{len(test_docs)})")
            
            # Create form data
            data = aiohttp.FormData()
            data.add_field('file', 
                          doc['content'].encode('utf-8'),
                          filename=doc['filename'],
                          content_type='text/plain')
            
            try:
                async with session.post('http://localhost:8001/process', data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"   ✅ Success: {result['organic_doc_id']}")
                        print(f"   🎯 Confidence: {result['confidence']:.1%}")
                        print(f"   📊 Staging: {result['ecosystem_status']['documents_in_staging']} docs")
                        print(f"   🌱 Seeds: {result['ecosystem_status']['collection_seeds']}")
                        print(f"   📚 Collections: {result['ecosystem_status']['mature_collections']}")
                    else:
                        print(f"   ❌ Error: {response.status}")
                        error_text = await response.text()
                        print(f"   Details: {error_text}")
                        
            except Exception as e:
                print(f"   ❌ Connection error: {e}")
        
        print(f"\n🔍 Getting final ecosystem status...")
        
        try:
            async with session.get('http://localhost:8001/ecosystem') as response:
                if response.status == 200:
                    ecosystem = await response.json()
                    print(f"\n📊 Final Ecosystem Status:")
                    print(f"   📥 Documents in staging: {ecosystem['ecosystem']['ecosystem_status']['documents_in_staging']}")
                    print(f"   🌱 Collection seeds: {ecosystem['ecosystem']['ecosystem_status']['collection_seeds']}")
                    print(f"   📚 Mature collections: {ecosystem['ecosystem']['ecosystem_status']['mature_collections']}")
                    print(f"   🏥 Health: {ecosystem['ecosystem']['health_summary']['health_percentage']:.1f}%")
                    
                    if ecosystem['suggestions']:
                        print(f"\n💡 AI Suggestions:")
                        for suggestion in ecosystem['suggestions'][:3]:
                            print(f"   • {suggestion['type']}: {suggestion['reason']}")
                
        except Exception as e:
            print(f"❌ Failed to get ecosystem status: {e}")
    
    print(f"\n✅ Test complete! Check http://localhost:8001/collections to see results")

if __name__ == "__main__":
    asyncio.run(test_organic_uploads())