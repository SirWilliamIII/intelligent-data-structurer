#!/usr/bin/env python3
"""
Test natural clustering by uploading similar documents
"""

import asyncio
import aiohttp

async def test_natural_clustering():
    """Upload groups of similar documents to trigger clustering."""
    
    print("🌱 Testing Natural Clustering Formation")
    print("=" * 50)
    
    # Group 1: Technical Documentation (should cluster)
    tech_docs = [
        {
            "filename": "docker-guide.md",
            "content": """# Docker Tutorial

## Basic Commands
```bash
docker build -t myapp .
docker run -p 3000:3000 myapp
docker ps
```

Learn containerization and deployment strategies."""
        },
        {
            "filename": "git-cheatsheet.md", 
            "content": """# Git Commands

## Basic Operations
```bash
git add .
git commit -m "message"
git push origin main
```

Version control essentials for developers."""
        },
        {
            "filename": "linux-commands.md",
            "content": """# Linux Terminal

## File Operations
```bash
ls -la
cd /home/user
chmod 755 file.sh
```

Essential command line operations."""
        },
        {
            "filename": "api-reference.md",
            "content": """# REST API Guide

## HTTP Methods
```
GET /api/users
POST /api/users
PUT /api/users/1
```

RESTful API design principles."""
        },
        {
            "filename": "devops-tools.md",
            "content": """# DevOps Toolkit

## CI/CD Pipeline
```yaml
stages:
  - build
  - test
  - deploy
```

Modern development workflow automation."""
        }
    ]
    
    # Group 2: Cooking Recipes (should cluster)
    cooking_docs = [
        {
            "filename": "banana-bread.md",
            "content": """# Banana Bread Recipe

## Ingredients
- 3 ripe bananas
- 1/3 cup melted butter
- 1 cup sugar
- 1 egg
- 1 tsp vanilla

## Instructions
1. Preheat oven to 350°F
2. Mash bananas
3. Mix all ingredients
4. Bake 60 minutes"""
        },
        {
            "filename": "pasta-sauce.md", 
            "content": """# Marinara Sauce

## Ingredients
- 2 cans crushed tomatoes
- 3 cloves garlic
- 1 onion diced
- 2 tsp oregano
- 1 tsp basil

## Instructions
1. Sauté onion and garlic
2. Add tomatoes and herbs
3. Simmer 30 minutes"""
        },
        {
            "filename": "beef-stew.md",
            "content": """# Hearty Beef Stew

## Ingredients
- 2 lbs beef chuck
- 4 carrots chopped
- 3 potatoes cubed
- 2 cups beef broth
- 1 tsp thyme

## Instructions
1. Brown beef in pot
2. Add vegetables and broth
3. Simmer 2 hours"""
        },
        {
            "filename": "apple-pie.md",
            "content": """# Classic Apple Pie

## Ingredients
- 6 apples sliced
- 1 cup sugar
- 2 tbsp flour
- 1 tsp cinnamon
- Pie crust

## Instructions
1. Mix apples with sugar and spices
2. Fill pie crust
3. Bake at 425°F for 45 minutes"""
        },
        {
            "filename": "chicken-curry.md",
            "content": """# Chicken Curry

## Ingredients
- 1 lb chicken thighs
- 1 can coconut milk
- 2 tbsp curry powder
- 1 onion
- 2 tsp ginger

## Instructions
1. Cook chicken until golden
2. Add spices and coconut milk
3. Simmer 25 minutes"""
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        
        print("\n📚 Uploading Technical Documentation Group...")
        for i, doc in enumerate(tech_docs):
            print(f"   📄 Uploading {doc['filename']} ({i+1}/{len(tech_docs)})")
            
            data = aiohttp.FormData()
            data.add_field('file', 
                          doc['content'].encode('utf-8'),
                          filename=doc['filename'],
                          content_type='text/markdown')
            
            try:
                async with session.post('http://localhost:8001/process', data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"      ✅ Staged: {result['ecosystem_status']['documents_in_staging']} docs")
                        print(f"      🌱 Seeds: {result['ecosystem_status']['collection_seeds']}")
                    else:
                        print(f"      ❌ Error: {response.status}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        print(f"\n🍳 Uploading Cooking Recipes Group...")
        for i, doc in enumerate(cooking_docs):
            print(f"   📄 Uploading {doc['filename']} ({i+1}/{len(cooking_docs)})")
            
            data = aiohttp.FormData()
            data.add_field('file',
                          doc['content'].encode('utf-8'), 
                          filename=doc['filename'],
                          content_type='text/markdown')
            
            try:
                async with session.post('http://localhost:8001/process', data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"      ✅ Staged: {result['ecosystem_status']['documents_in_staging']} docs")
                        print(f"      🌱 Seeds: {result['ecosystem_status']['collection_seeds']}")
                        if result['ecosystem_status']['collection_seeds'] > 0:
                            print(f"      🎉 SEED FORMATION DETECTED!")
                    else:
                        print(f"      ❌ Error: {response.status}")
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        # Trigger evolution
        print(f"\n🧬 Triggering Evolution...")
        try:
            async with session.post('http://localhost:8001/evolve') as response:
                if response.status == 200:
                    result = await response.json()
                    ecosystem = result['ecosystem_after']
                    print(f"   📊 After Evolution:")
                    print(f"      📥 Staging: {ecosystem['ecosystem_status']['documents_in_staging']}")
                    print(f"      🌱 Seeds: {ecosystem['ecosystem_status']['collection_seeds']}")
                    print(f"      📚 Mature Collections: {ecosystem['ecosystem_status']['mature_collections']}")
                    
                    if ecosystem['collection_sizes']:
                        print(f"      🎉 COLLECTIONS BORN:")
                        for name, size in ecosystem['collection_sizes'].items():
                            print(f"         • {name}: {size} documents")
        except Exception as e:
            print(f"   ❌ Evolution error: {e}")
        
        # Final status
        print(f"\n📊 Final Ecosystem Check...")
        try:
            async with session.get('http://localhost:8001/collections') as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   📚 Total Collections: {len(result['collections'])}")
                    for collection in result['collections']:
                        print(f"      • {collection['name']}: {collection['document_count']} docs ({collection['type']})")
                        if 'themes' in collection:
                            print(f"        Themes: {', '.join(collection['themes'][:3])}")
        except Exception as e:
            print(f"   ❌ Status error: {e}")
    
    print(f"\n✅ Clustering test complete!")
    print(f"🌐 View results: http://localhost:8001/collections")

if __name__ == "__main__":
    asyncio.run(test_natural_clustering())