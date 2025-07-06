"""
Database management and table creation for structured data storage.
"""

import asyncio
import json
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Text, Float, Boolean, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from .config import settings
from .classifier import ContentType

Base = declarative_base()

class DatabaseManager:
    """Manages database connections and table creation."""
    
    def __init__(self):
        print(f"Database URL: {settings.database_url}")
        self.engine = create_async_engine(settings.database_url, echo=True)
        self.session_factory = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.metadata = MetaData()
        self.created_tables = set()
    
    async def create_table_for_content_type(self, content_type: ContentType) -> str:
        """Create appropriate table for the detected content type."""
        
        table_schemas = {
            ContentType.CONTACT_INFO: {
                'table_name': 'contacts',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('name', String(255)),
                    Column('email', String(255)),
                    Column('phone', String(50)),
                    Column('mobile', String(50)),
                    Column('address', Text),
                    Column('company', String(255)),
                    Column('title', String(255)),
                    Column('department', String(100)),
                    Column('notes', Text),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255)),
                    Column('extracted_at', String(50))
                ]
            },
            
            ContentType.BUSINESS_CARD: {
                'table_name': 'business_cards',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('name', String(255)),
                    Column('title', String(255)),
                    Column('company', String(255)),
                    Column('email', String(255)),
                    Column('phone', String(50)),
                    Column('website', String(255)),
                    Column('address', Text),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255))
                ]
            },
            
            ContentType.PRODUCT_DATA: {
                'table_name': 'products',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('name', String(500)),
                    Column('sku', String(100)),
                    Column('price', Float),
                    Column('category', String(255)),
                    Column('description', Text),
                    Column('in_stock', Integer),
                    Column('supplier', String(255)),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255))
                ]
            },
            
            ContentType.EVENT_INFO: {
                'table_name': 'events',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('title', String(500)),
                    Column('event_date', String(50)),  # Store as string initially
                    Column('event_time', String(50)),
                    Column('location', String(500)),
                    Column('address', Text),
                    Column('organizer', String(255)),
                    Column('capacity', Integer),
                    Column('description', Text),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255))
                ]
            },
            
            ContentType.RECIPE: {
                'table_name': 'recipes',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('title', String(500)),
                    Column('prep_time', String(50)),
                    Column('cook_time', String(50)),
                    Column('total_time', String(50)),
                    Column('servings', Integer),
                    Column('difficulty', String(50)),
                    Column('cuisine', String(100)),
                    Column('ingredients', JSON),  # Store as JSON array
                    Column('instructions', JSON),  # Store as JSON array
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255))
                ]
            },
            
            ContentType.FINANCIAL_DATA: {
                'table_name': 'transactions',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('transaction_id', String(100)),
                    Column('date', String(50)),
                    Column('amount', Float),
                    Column('description', Text),
                    Column('account', String(100)),
                    Column('type', String(50)),
                    Column('category', String(100)),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255))
                ]
            },
            
            ContentType.EMPLOYEE_DATA: {
                'table_name': 'employees',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('employee_id', String(50)),
                    Column('name', String(255)),
                    Column('email', String(255)),
                    Column('department', String(100)),
                    Column('position', String(255)),
                    Column('salary', Float),
                    Column('hire_date', String(50)),
                    Column('manager', String(255)),
                    Column('phone', String(50)),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255))
                ]
            },
            
            ContentType.ARTICLE: {
                'table_name': 'articles',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('title', String(500)),
                    Column('author', String(255)),
                    Column('publish_date', String(50)),
                    Column('content', Text),
                    Column('summary', Text),
                    Column('tags', JSON),
                    Column('word_count', Integer),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255))
                ]
            }
        }
        
        # Get schema for content type
        schema = table_schemas.get(content_type)
        if not schema:
            # Create generic table for unknown types
            schema = {
                'table_name': f'{content_type.value}_data',
                'columns': [
                    Column('id', Integer, primary_key=True),
                    Column('content', Text),
                    Column('extracted_data', JSON),
                    Column('created_at', DateTime, default=datetime.utcnow),
                    Column('source_file', String(255)),
                    Column('extracted_at', String(50))
                ]
            }
        
        table_name = schema['table_name']
        
        # Skip if already created
        if table_name in self.created_tables:
            return table_name
        
        # Create table
        try:
            async with self.engine.begin() as conn:
                # Check if table exists
                result = await conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table_name}'
                    );
                """))
                
                table_exists = result.scalar()
                
                if not table_exists:
                    # Build CREATE TABLE statement
                    columns_sql = []
                    for col in schema['columns']:
                        col_sql = f"{col.name} {self._get_postgres_type(col.type)}"
                        if col.primary_key:
                            col_sql += " PRIMARY KEY"
                        if hasattr(col, 'default') and col.default:
                            if col.name == 'created_at':
                                col_sql += " DEFAULT CURRENT_TIMESTAMP"
                        columns_sql.append(col_sql)
                    
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        {', '.join(columns_sql)}
                    );
                    """
                    
                    await conn.execute(text(create_sql))
                    logger.info(f"Created table: {table_name}")
                else:
                    logger.info(f"Table already exists: {table_name}")
                
                self.created_tables.add(table_name)
                return table_name
                
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise
    
    def _get_postgres_type(self, sqlalchemy_type) -> str:
        """Convert SQLAlchemy types to PostgreSQL types."""
        type_mapping = {
            'INTEGER': 'INTEGER',
            'VARCHAR': 'VARCHAR(255)',
            'TEXT': 'TEXT',
            'FLOAT': 'FLOAT',
            'BOOLEAN': 'BOOLEAN',
            'DATETIME': 'TIMESTAMP',
            'JSON': 'JSONB'
        }
        
        type_str = str(sqlalchemy_type)
        if 'VARCHAR' in type_str:
            return type_str.replace('VARCHAR', 'VARCHAR')
        
        for sa_type, pg_type in type_mapping.items():
            if sa_type in type_str:
                return pg_type
        
        return 'TEXT'  # Default fallback
    
    async def insert_structured_data(self, table_name: str, data: Dict[str, Any]) -> int:
        """Insert structured data into the specified table."""
        
        try:
            async with self.session_factory() as session:
                # Convert JSON fields to strings for proper insertion
                processed_data = {}
                for key, value in data.items():
                    if isinstance(value, dict) or isinstance(value, list):
                        processed_data[key] = json.dumps(value)
                    elif isinstance(value, set):
                        processed_data[key] = json.dumps(list(value))
                    else:
                        processed_data[key] = value
                
                # Build INSERT statement
                columns = list(processed_data.keys())
                
                placeholders = ', '.join([f':{col}' for col in columns])
                columns_str = ', '.join(columns)
                
                sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
                RETURNING id;
                """
                
                result = await session.execute(text(sql), processed_data)
                row_id = result.scalar()
                await session.commit()
                
                logger.info(f"Inserted data into {table_name}, ID: {row_id}")
                return row_id
                
        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {e}")
            raise
    
    async def get_table_data(self, table_name: str, limit: int = 100) -> List[Dict]:
        """Get data from a table."""
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(text(f"SELECT * FROM {table_name} ORDER BY created_at DESC LIMIT {limit}"))
                rows = result.fetchall()
                columns = result.keys()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get data from {table_name}: {e}")
            return []
    
    async def list_tables(self) -> List[str]:
        """List all tables created by the system."""
        
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name != 'alembic_version'
                    ORDER BY table_name;
                """))
                
                return [row[0] for row in result.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []
    
    async def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        
        tables = await self.list_tables()
        stats = {}
        
        for table in tables:
            try:
                async with self.session_factory() as session:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    stats[table] = count
            except Exception as e:
                logger.error(f"Failed to get stats for {table}: {e}")
                stats[table] = 0
        
        return stats
    
    async def create_intelligent_table(self, table_name: str, schema_evolution) -> str:
        """Create table with intelligent schema based on content analysis."""
        
        try:
            async with self.engine.begin() as conn:
                # Check if table exists
                result = await conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{table_name}'
                    );
                """))
                
                table_exists = result.scalar()
                
                if not table_exists:
                    # Create base schema with intelligent additions
                    base_columns = [
                        "id SERIAL PRIMARY KEY",
                        "content TEXT",
                        "extracted_at VARCHAR(50)",
                        "source_file VARCHAR(255)",
                        "priority FLOAT DEFAULT 5.0",
                        "domain VARCHAR(50)",
                        "content_type VARCHAR(50)"
                    ]
                    
                    # Add suggested columns from schema evolution
                    for col_name, col_type in schema_evolution.suggested_additions.items():
                        if col_name not in ['id', 'content', 'extracted_at', 'source_file']:
                            base_columns.append(f"{col_name} {col_type}")
                    
                    # Add common intelligent fields
                    intelligent_columns = [
                        "entities JSONB",
                        "emails JSONB",
                        "phones JSONB", 
                        "dates JSONB",
                        "urls JSONB",
                        "word_count INTEGER",
                        "char_count INTEGER",
                        "original_char_count INTEGER",
                        "semantic_signature JSONB",
                        "content_markers JSONB",
                        "structural_patterns JSONB",
                        "content_specific_data JSONB"
                    ]
                    
                    all_columns = base_columns + intelligent_columns
                    
                    create_sql = f"""
                    CREATE TABLE {table_name} (
                        {', '.join(all_columns)}
                    );
                    """
                    
                    await conn.execute(text(create_sql))
                    logger.info(f"Created intelligent table: {table_name}")
                    
                    # Create indexes for better performance (one at a time)
                    indexes = [
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_priority ON {table_name}(priority DESC)",
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_domain ON {table_name}(domain)",
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_content_type ON {table_name}(content_type)",
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_extracted_at ON {table_name}(extracted_at DESC)"
                    ]
                    
                    for index_sql in indexes:
                        await conn.execute(text(index_sql))
                    
                    logger.info(f"Created indexes for table: {table_name}")
                    
                else:
                    logger.info(f"Intelligent table already exists: {table_name}")
                    
                    # TODO: In a more advanced system, we could check for schema evolution
                    # and add new columns if they don't exist
                
                self.created_tables.add(table_name)
                return table_name
                
        except Exception as e:
            logger.error(f"Failed to create intelligent table {table_name}: {e}")
            raise

# DatabaseManager class is available for import
