#!/usr/bin/env python3
"""
Simple script to view document summaries without all the content clutter.
"""

import asyncpg
import asyncio
from datetime import datetime

async def view_documents():
    """View document summaries in a clean format."""
    
    # Connect to database
    conn = await asyncpg.connect("postgresql://will@localhost:5432/intelligent_data")
    
    print("🗂️  INTELLIGENT DOCUMENT PROCESSOR - SUMMARY VIEW")
    print("=" * 60)
    
    # Get all tables
    tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name != 'alembic_version'
        ORDER BY table_name;
    """
    
    tables = await conn.fetch(tables_query)
    
    total_docs = 0
    
    for table_row in tables:
        table_name = table_row['table_name']
        
        # Get documents from this table
        try:
            docs_query = f"""
                SELECT 
                    source_file,
                    char_count,
                    word_count,
                    priority,
                    COALESCE(domain, '') as domain,
                    COALESCE(content_type, '') as content_type,
                    extracted_at
                FROM {table_name}
                ORDER BY id DESC
                LIMIT 20;
            """
            
            docs = await conn.fetch(docs_query)
            
            if docs:
                print(f"\n📋 {table_name.upper()} ({len(docs)} documents)")
                print("-" * 60)
                
                for doc in docs:
                    total_docs += 1
                    
                    # Format extracted time
                    extracted_time = doc['extracted_at']
                    if isinstance(extracted_time, str):
                        try:
                            extracted_time = datetime.fromisoformat(extracted_time.replace('Z', '+00:00'))
                            time_str = extracted_time.strftime('%m/%d %H:%M')
                        except:
                            time_str = extracted_time[:10] if len(extracted_time) > 10 else extracted_time
                    else:
                        time_str = str(extracted_time)[:10]
                    
                    # Clean filename
                    filename = doc['source_file']
                    if len(filename) > 35:
                        filename = filename[:32] + "..."
                    
                    # Show compact summary
                    domain = doc['domain'] or 'general'
                    content_type = doc['content_type'] or 'doc'
                    
                    print(f"📄 {filename:<35} | 📊 {doc['word_count']:>4}w {doc['char_count']:>5}c | 🎯 P{doc['priority']} | 🏷️ {domain}/{content_type} | ⏰ {time_str}")
                    
        except Exception as e:
            print(f"❌ Error reading {table_name}: {e}")
    
    print("=" * 60)
    print(f"📈 TOTAL: {total_docs} documents processed across {len(tables)} tables")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(view_documents())