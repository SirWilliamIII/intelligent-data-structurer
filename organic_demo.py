#!/usr/bin/env python3
"""
Organic Collections Demo - Show the brilliant evolution in action
"""

import asyncio
import pymongo
from datetime import datetime, timedelta
from core.organic_collections import OrganicCollectionManager
from core.intelligent_analyzer import IntelligentAnalyzer
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live

console = Console()

class OrganicDemo:
    """Demonstrate the organic collection system in action."""
    
    def __init__(self):
        # Connect to MongoDB
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.db = self.client['intelligent_data']
        
        # Initialize systems
        self.organic_manager = OrganicCollectionManager(self.client)
        self.analyzer = IntelligentAnalyzer()
        
        # Demo state
        self.demo_docs_processed = 0
        self.demo_start_time = None
    
    async def run_full_demo(self):
        """Run the complete organic collections demo."""
        
        console.print(Panel.fit("🧠 Organic Collection Evolution Demo", style="bold green"))
        console.print("\n🌱 This system treats collections like living organisms that:")
        console.print("   • Start as seeds when similar documents cluster together")
        console.print("   • Grow by attracting related content")
        console.print("   • Evolve their understanding over time")
        console.print("   • Merge when they overlap, split when they become too broad")
        console.print("   • Self-organize into natural knowledge clusters\n")
        
        # Step 1: Show initial ecosystem state
        await self.show_ecosystem_status()
        
        # Step 2: Process existing documents through organic system
        console.print("\n🔄 Migrating existing documents to organic system...")
        await self.migrate_existing_documents()
        
        # Step 3: Show evolution in action
        await self.show_evolution_process()
        
        # Step 4: Demonstrate intelligent queries
        await self.demonstrate_intelligent_queries()
        
        # Step 5: Show ecosystem insights
        await self.show_final_insights()
    
    async def show_ecosystem_status(self):
        """Show the current state of the ecosystem."""
        
        insights = await self.organic_manager.get_collection_insights()
        
        table = Table(title="🌍 Ecosystem Status", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Documents in Staging", str(insights['ecosystem_status']['documents_in_staging']))
        table.add_row("Collection Seeds", str(insights['ecosystem_status']['collection_seeds']))
        table.add_row("Mature Collections", str(insights['ecosystem_status']['mature_collections']))
        table.add_row("Organized Documents", str(insights['ecosystem_status']['total_organized_documents']))
        
        console.print(table)
    
    async def migrate_existing_documents(self):
        """Migrate existing documents through the organic system."""
        
        # Get all existing collections (except staging and registry)
        exclude_collections = ['document_staging', 'collection_registry', 'collection_seeds']
        existing_collections = [name for name in self.db.list_collection_names() 
                              if name not in exclude_collections]
        
        console.print(f"\n📦 Found {len(existing_collections)} existing collections to migrate")
        
        total_docs = 0
        for collection_name in existing_collections:
            count = self.db[collection_name].count_documents({})
            total_docs += count
            console.print(f"   • {collection_name}: {count} documents")
        
        if total_docs == 0:
            console.print("⚠️  No existing documents found. Upload some documents first!")
            return
        
        # Process documents through organic system
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("🔄 Processing documents...", total=total_docs)
            
            processed = 0
            for collection_name in existing_collections:
                docs = list(self.db[collection_name].find())
                
                for doc in docs:
                    # Clean up the document (remove MongoDB ObjectId)
                    doc_data = {k: v for k, v in doc.items() if k != '_id'}
                    
                    # Add missing fields if needed
                    if 'semantic_signature' not in doc_data:
                        doc_data['semantic_signature'] = {
                            'domain_keywords': [],
                            'structural_patterns': [],
                            'entity_types': [],
                            'content_markers': [],
                            'semantic_hash': 'legacy'
                        }
                    
                    # Process through organic system
                    await self.organic_manager.process_document(doc_data)
                    
                    processed += 1
                    progress.update(task, advance=1)
                    
                    # Small delay to show progress
                    if processed % 5 == 0:
                        await asyncio.sleep(0.1)
                
                # Clear the old collection after migration
                self.db[collection_name].drop()
        
        console.print(f"✅ Migrated {processed} documents to organic system")
    
    async def show_evolution_process(self):
        """Show the evolution process happening in real-time."""
        
        console.print("\n🧬 Watching Evolution in Real-Time...")
        
        # Trigger evolution cycles
        for cycle in range(3):
            console.print(f"\n🔄 Evolution Cycle {cycle + 1}")
            
            # Force some evolution
            await self.organic_manager._evolve_collections()
            await self.organic_manager._health_check_collections()
            
            # Show what happened
            insights = await self.organic_manager.get_collection_insights()
            
            console.print(f"   📊 Staging: {insights['ecosystem_status']['documents_in_staging']} docs")
            console.print(f"   🌱 Seeds: {insights['ecosystem_status']['collection_seeds']}")
            console.print(f"   📚 Mature Collections: {insights['ecosystem_status']['mature_collections']}")
            
            # Show collection sizes
            if insights['collection_sizes']:
                console.print("   📂 Collection Growth:")
                for name, size in insights['collection_sizes'].items():
                    console.print(f"      • {name}: {size} documents")
            
            await asyncio.sleep(1)
    
    async def demonstrate_intelligent_queries(self):
        """Demonstrate intelligent cross-collection queries."""
        
        console.print("\n🔍 Intelligent Cross-Collection Queries")
        
        # Get insights about the evolved ecosystem
        insights = await self.organic_manager.get_collection_insights()
        
        if not insights['collection_sizes']:
            console.print("⚠️  No mature collections found yet. Evolution needs more time!")
            return
        
        # Query 1: Show collection DNA
        console.print("\n🧬 Collection DNA Analysis:")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Collection", style="cyan")
        table.add_column("Documents", style="green")
        table.add_column("Core Themes", style="yellow")
        table.add_column("Health", style="red")
        
        for collection_name, doc_count in insights['collection_sizes'].items():
            # Get collection DNA
            if collection_name not in self.organic_manager.collection_dna:
                await self.organic_manager._analyze_collection_dna(collection_name)
            
            dna = self.organic_manager.collection_dna.get(collection_name)
            health = self.organic_manager.collection_health.get(collection_name)
            
            themes = list(dna.core_themes)[:3] if dna else ["Unknown"]
            health_score = f"{health.coherence_score:.1%}" if health else "Unknown"
            
            table.add_row(
                collection_name,
                str(doc_count),
                ", ".join(themes),
                health_score
            )
        
        console.print(table)
        
        # Query 2: Find documents across collections
        console.print("\n🔍 Cross-Collection Intelligence:")
        
        # Demo query: Find all technical documents
        tech_collections = [name for name in insights['collection_sizes'].keys() 
                          if 'technical' in name.lower() or 'kubernetes' in name.lower()]
        
        if tech_collections:
            total_tech_docs = sum(self.db[name].count_documents({}) for name in tech_collections)
            console.print(f"   🔧 Technical Documents: {total_tech_docs} across {len(tech_collections)} collections")
            
            for collection in tech_collections:
                count = self.db[collection].count_documents({})
                console.print(f"      • {collection}: {count} docs")
        
        # Demo query: Find documents with contact info
        contact_docs = 0
        for collection_name in insights['collection_sizes'].keys():
            collection = self.db[collection_name]
            count = collection.count_documents({
                "$or": [
                    {"emails": {"$ne": [], "$exists": True}},
                    {"phones": {"$ne": [], "$exists": True}}
                ]
            })
            contact_docs += count
        
        console.print(f"   📞 Documents with Contact Info: {contact_docs} across all collections")
    
    async def show_final_insights(self):
        """Show final insights about the evolved ecosystem."""
        
        console.print("\n📊 Final Ecosystem Analysis")
        
        # Get comprehensive insights
        insights = await self.organic_manager.get_collection_insights()
        suggestions = await self.organic_manager.suggest_collection_improvements()
        
        # Ecosystem health
        health_panel = Panel(
            f"""📈 Health Score: {insights['health_summary']['health_percentage']:.1f}%
🌱 Seeds Ready: {insights['seeds_ready_for_birth']}
📚 Total Collections: {insights['health_summary']['total_collections']}
🏥 Healthy Collections: {insights['health_summary']['healthy_collections']}""",
            title="🌍 Ecosystem Health",
            style="green"
        )
        
        console.print(health_panel)
        
        # Suggestions
        if suggestions:
            console.print("\n💡 AI Suggestions for Improvement:")
            for suggestion in suggestions[:3]:  # Show top 3
                console.print(f"   • {suggestion['type'].replace('_', ' ').title()}: {suggestion['reason']}")
        
        # Show the evolution tree
        await self.show_evolution_tree(insights)
    
    async def show_evolution_tree(self, insights):
        """Show the evolution as a tree structure."""
        
        tree = Tree("🌳 Collection Evolution Tree")
        
        # Staging branch
        staging_branch = tree.add(f"📥 Staging ({insights['ecosystem_status']['documents_in_staging']} docs)")
        
        # Seeds branch
        seeds_branch = tree.add(f"🌱 Seeds ({insights['ecosystem_status']['collection_seeds']} growing)")
        
        # Mature collections branch
        mature_branch = tree.add(f"📚 Mature Collections ({insights['ecosystem_status']['mature_collections']})")
        
        for collection_name, doc_count in insights['collection_sizes'].items():
            # Get health info
            health = self.organic_manager.collection_health.get(collection_name)
            health_emoji = "💚" if health and health.coherence_score > 0.6 else "💛" if health else "❓"
            
            collection_node = mature_branch.add(f"{health_emoji} {collection_name} ({doc_count} docs)")
            
            # Add themes if available
            dna = self.organic_manager.collection_dna.get(collection_name)
            if dna and dna.core_themes:
                themes = list(dna.core_themes)[:3]
                collection_node.add(f"🏷️  Themes: {', '.join(themes)}")
        
        console.print(tree)
        
        # Final summary
        console.print(Panel.fit(
            """🎉 Organic Evolution Complete!

Your documents have naturally organized themselves into meaningful clusters.
Each collection represents a genuine theme discovered in your content.
The system continues to learn and evolve as you add more documents.

Key Benefits:
• No hardcoded categories - collections emerge from actual patterns
• Automatic quality control through health monitoring  
• Self-healing system that merges/splits collections as needed
• Intelligent cross-collection search and analysis""",
            title="✨ Evolution Success",
            style="bold green"
        ))

async def main():
    """Run the organic collections demo."""
    
    demo = OrganicDemo()
    
    try:
        await demo.run_full_demo()
    except KeyboardInterrupt:
        console.print("\n👋 Demo interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check dependencies
    try:
        import rich
        import sklearn
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install rich scikit-learn numpy")
        exit(1)
    
    asyncio.run(main())