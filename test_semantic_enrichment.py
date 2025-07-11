#!/usr/bin/env python3
"""
Test the new semantic enrichment system with NFL schedule data
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from core.intelligent_analyzer import IntelligentAnalyzer

async def test_nfl_enrichment():
    """Test if the system can organically discover that NFL teams are sports-related."""
    
    analyzer = IntelligentAnalyzer()
    
    # Sample NFL schedule content (like what's in the database)
    test_content = """Arizona Cardinals - 2025 Mock Schedule
Week 1: vs Atlanta Falcons
Week 2: vs Baltimore Ravens  
Week 3: vs Buffalo Bills
Week 4: vs Carolina Panthers
Week 5: vs Chicago Bears
Week 6: vs Cincinnati Bengals
Week 7: vs Cleveland Browns
Week 8: vs Dallas Cowboys"""

    filename = "Arizona_Cardinals_schedule.pdf"
    
    print("🧪 Testing semantic enrichment on NFL schedule...")
    print(f"📄 Content: {test_content[:100]}...")
    
    try:
        result = await analyzer.analyze_content(test_content, filename)
        
        print(f"\n📊 Analysis Results:")
        print(f"   Collection: {result.table_name}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Reasoning: {result.reasoning}")
        
        # Check semantic signature
        semantic_sig = result.extracted_data.get('semantic_signature', {})
        keywords = semantic_sig.get('domain_keywords', [])
        
        print(f"\n🔍 Discovered Keywords:")
        for i, keyword in enumerate(keywords[:10]):
            print(f"   {i+1}. {keyword}")
        
        # Check if sports domain was discovered
        sports_keywords = ['nfl', 'football', 'team', 'sport', 'league', 'professional']
        found_sports = [kw for kw in keywords if kw in sports_keywords]
        
        if found_sports:
            print(f"\n✅ SUCCESS: Discovered sports domain keywords: {found_sports}")
        else:
            print(f"\n❌ ISSUE: No sports domain keywords found")
            print(f"   Top keywords were: {keywords[:5]}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_nfl_enrichment())