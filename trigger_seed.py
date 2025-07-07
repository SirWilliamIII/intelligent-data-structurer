#!/usr/bin/env python3
"""
Upload one more similar document to trigger seed creation
"""

import asyncio
import aiohttp

async def trigger_seed_formation():
    """Upload a document that should trigger seed formation."""
    
    print("🌱 Triggering Seed Formation")
    print("=" * 40)
    
    # Add one more recipe that should push us over the edge
    trigger_doc = {
        "filename": "brownies.md",
        "content": """# Fudgy Brownies

## Ingredients
- 1 cup butter
- 2 cups sugar
- 4 eggs
- 1 cup flour
- 1/2 cup cocoa powder

## Instructions
1. Preheat oven to 350°F
2. Melt butter and mix with sugar
3. Add eggs one at a time
4. Fold in flour and cocoa
5. Bake 25 minutes"""
    }
    
    async with aiohttp.ClientSession() as session:
        print(f"📄 Uploading {trigger_doc['filename']}...")
        
        data = aiohttp.FormData()
        data.add_field('file', 
                      trigger_doc['content'].encode('utf-8'),
                      filename=trigger_doc['filename'],
                      content_type='text/markdown')
        
        try:
            async with session.post('http://localhost:8001/process', data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Success!")
                    print(f"   📊 Staging: {result['ecosystem_status']['documents_in_staging']}")
                    print(f"   🌱 Seeds: {result['ecosystem_status']['collection_seeds']}")
                    print(f"   📚 Collections: {result['ecosystem_status']['mature_collections']}")
                    print(f"   🎯 Confidence: {result['confidence']:.1%}")
                    
                    if result['ecosystem_status']['collection_seeds'] > 0:
                        print(f"   🎉 SEED FORMED!")
                else:
                    print(f"❌ Error: {response.status}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Force evolution multiple times
        for i in range(3):
            print(f"\n🧬 Evolution cycle {i+1}...")
            try:
                async with session.post('http://localhost:8001/evolve') as response:
                    if response.status == 200:
                        result = await response.json()
                        ecosystem = result['ecosystem_after']
                        print(f"   📊 Staging: {ecosystem['ecosystem_status']['documents_in_staging']}")
                        print(f"   🌱 Seeds: {ecosystem['ecosystem_status']['collection_seeds']}")
                        print(f"   📚 Collections: {ecosystem['ecosystem_status']['mature_collections']}")
                        
                        if ecosystem['collection_sizes']:
                            print(f"   🎉 COLLECTIONS BORN:")
                            for name, size in ecosystem['collection_sizes'].items():
                                print(f"      • {name}: {size} documents")
                        
                        if ecosystem['ecosystem_status']['collection_seeds'] > 0:
                            print(f"   🌱 Seeds detected!")
                        if ecosystem['ecosystem_status']['mature_collections'] > 0:
                            print(f"   📚 Collections detected!")
                            
            except Exception as e:
                print(f"   ❌ Evolution error: {e}")
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(trigger_seed_formation())