#!/usr/bin/env python3
"""
Database Reset for Fresh Start

Reset the database completely and initialize the new intelligent system.
This implements the user's requirement for a fresh database to test
the intelligent taxonomy system properly.
"""

import asyncio
import sys
import os
import pymongo
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(__file__))

from core.intelligent_collection_engine import IntelligentCollectionEngine
from loguru import logger

async def reset_database_fresh_start():
    """
    Complete database reset for fresh start testing.
    
    User requirement: "let's start with a fresh database and reupload the documents"
    """
    
    print("🧠 INTELLIGENT TAXONOMY SYSTEM - DATABASE RESET")
    print("=" * 55)
    print()
    print("🎯 Priorities: Intelligence and Accuracy over Speed")
    print("📊 Volume-based triggers: 10 (small) / 50 (medium) / 100 (large)")
    print("🤖 Claude-powered semantic analysis")
    print("🔄 Concept drift detection active")
    print("🌟 Ready for mindblowingly intelligent document organization!")
    print()
    
    # Connect to MongoDB
    try:
        client = pymongo.MongoClient('mongodb://localhost:27017')
        db = client['intelligent_data']
        
        # Test connection
        client.admin.command('ping')
        print("✅ Connected to MongoDB successfully")
        
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("💡 Make sure MongoDB is running: brew services start mongodb-community")
        return False
    
    # Initialize intelligent collection engine
    print("\n🧠 Initializing Intelligent Collection Engine...")
    engine = IntelligentCollectionEngine(client)
    
    # Perform fresh database reset
    print("\n🗑️ Performing fresh database reset...")
    
    # Show what we're about to delete
    collection_names = db.list_collection_names()
    if collection_names:
        print(f"📋 Collections to be removed: {len(collection_names)}")
        for name in collection_names:
            count = db[name].count_documents({})
            print(f"   • {name}: {count} documents")
    else:
        print("📋 Database is already empty")
    
    # Confirm reset
    print(f"\n⚠️  This will DELETE ALL EXISTING DATA!")
    print("✨ But we'll get a fresh, intelligent taxonomy system!")
    print("\n🚀 Proceeding with fresh database reset (auto-confirmed)...")
    
    # Perform the reset
    print("\n🚀 Performing fresh reset...")
    success = await engine.reset_database()
    
    if success:
        print("✅ Database reset successful!")
        
        # Validate system health
        print("\n🏥 Validating system health...")
        health_report = await engine.validate_system_health()
        
        print(f"Overall Status: {health_report['overall_status']}")
        for component, status in health_report['components'].items():
            print(f"   • {component}: {status}")
        
        if health_report['issues']:
            print("\n⚠️  Issues detected:")
            for issue in health_report['issues']:
                print(f"   • {issue}")
        else:
            print("✅ All systems operational!")
        
        # Show final state
        print("\n📊 Final Database State:")
        collections = db.list_collection_names()
        if collections:
            for name in collections:
                count = db[name].count_documents({})
                print(f"   • {name}: {count} documents")
        else:
            print("   • Empty database - ready for intelligent processing!")
        
        print("\n🎉 FRESH START COMPLETE!")
        print("=" * 30)
        print("🚀 Ready to upload documents for intelligent processing")
        print("📄 Upload your NFL schedules, NBA schedules, or contracts")
        print("🧠 Watch the system create intelligent collections organically")
        print("🌟 No more 'chicago_bulls_game_64_documents'!")
        print("✨ Only semantically meaningful collections!")
        
        return True
        
    else:
        print("❌ Database reset failed!")
        return False

def show_next_steps():
    """Show user what to do next."""
    print("\n📋 NEXT STEPS:")
    print("=" * 20)
    print("1. Start the server: python main.py")
    print("2. Upload documents through the web interface")
    print("3. Watch intelligent collections emerge!")
    print()
    print("🔍 Monitor the process:")
    print("   • Check logs for Claude analysis")
    print("   • Watch concept drift detection")
    print("   • See taxonomy evolution in real-time")
    print()
    print("🎯 Expected Results:")
    print("   • NFL schedules → 'nfl_team_schedules' (not game-specific!)")
    print("   • NBA schedules → 'nba_team_schedules' or 'sports_schedules'")
    print("   • Contracts → 'sports_contracts' (separate from schedules)")
    print("   • Automatic hierarchy: sports → nfl/nba → specific teams")

if __name__ == "__main__":
    try:
        # Run the reset
        success = asyncio.run(reset_database_fresh_start())
        
        if success:
            show_next_steps()
        else:
            print("\n💡 Troubleshooting:")
            print("   • Ensure MongoDB is running")
            print("   • Check network connectivity")
            print("   • Verify database permissions")
            
    except KeyboardInterrupt:
        print("\n⏹️ Reset cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()