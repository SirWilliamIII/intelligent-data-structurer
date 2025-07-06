"""
Database management using MongoDB for document storage.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from .config import settings
from .classifier import ContentType
from .mongo_database import mongo_manager

class DatabaseManager:
    """Manages database connections and table creation."""
    
    def __init__(self):
        self.mongo_manager = mongo_manager
        self.connected = False
    
    async def create_table_for_content_type(self, content_type: ContentType) -> str:
        """Create appropriate collection for the detected content type."""
        
        # Map content types to collection names
        collection_mapping = {
            ContentType.CONTACT_INFO: 'contacts',
            ContentType.BUSINESS_CARD: 'business_cards',
            ContentType.PRODUCT_DATA: 'products',
            ContentType.EVENT_INFO: 'events',
            ContentType.RECIPE: 'cooking_recipes',
            ContentType.FINANCIAL_DATA: 'transactions',
            ContentType.EMPLOYEE_DATA: 'employees',
            ContentType.ARTICLE: 'articles'
        }
        
        collection_name = collection_mapping.get(content_type, f'{content_type.value}_documents')
        
        # Ensure MongoDB connection
        await self._ensure_connected()
        
        # Get collection (will create indexes automatically)
        await self.mongo_manager.get_collection(collection_name)
        
        logger.info(f"Collection ready: {collection_name}")
        return collection_name
    
    async def _ensure_connected(self):
        """Ensure MongoDB connection is established."""
        if not self.connected:
            await self.mongo_manager.connect()
            self.connected = True
    
    async def insert_structured_data(self, collection_name: str, data: Dict[str, Any]) -> str:
        """Insert structured data into the specified collection."""
        
        try:
            await self._ensure_connected()
            
            # Insert document into MongoDB
            doc_id = await self.mongo_manager.insert_document(collection_name, data)
            
            logger.info(f"Inserted data into {collection_name}, ID: {doc_id}")
            return doc_id
                
        except Exception as e:
            logger.error(f"Failed to insert data into {collection_name}: {e}")
            raise
    
    async def get_table_data(self, collection_name: str, limit: int = 100) -> List[Dict]:
        """Get data from a collection."""
        
        try:
            await self._ensure_connected()
            
            documents = await self.mongo_manager.find_documents(collection_name, limit=limit)
            return documents
                
        except Exception as e:
            logger.error(f"Failed to get data from {collection_name}: {e}")
            return []
    
    async def list_tables(self) -> List[str]:
        """List all collections in the database."""
        
        try:
            await self._ensure_connected()
            
            collections = await self.mongo_manager.list_collections()
            return collections
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def get_table_stats(self) -> Dict[str, int]:
        """Get document counts for all collections."""
        
        try:
            await self._ensure_connected()
            
            stats = await self.mongo_manager.get_collection_stats()
            return stats
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def create_intelligent_collection(self, table_name: str, schema_evolution) -> str:
        """Create intelligent collection with schema evolution."""
        
        try:
            await self._ensure_connected()
            
            # Use the table name as collection name
            collection_name = table_name
            
            # Get collection (will create indexes automatically)
            await self.mongo_manager.get_collection(collection_name)
            
            logger.info(f"Intelligent collection ready: {collection_name}")
            return collection_name
                
        except Exception as e:
            logger.error(f"Failed to create intelligent collection {table_name}: {e}")
            raise
    
    async def insert_document(self, collection_name: str, data: Dict[str, Any]) -> str:
        """Insert document into the specified collection."""
        
        try:
            await self._ensure_connected()
            
            # Insert document into MongoDB
            doc_id = await self.mongo_manager.insert_document(collection_name, data)
            
            logger.info(f"Inserted document into {collection_name}, ID: {doc_id}")
            return doc_id
                
        except Exception as e:
            logger.error(f"Failed to insert document into {collection_name}: {e}")
            raise

# DatabaseManager class is available for import
