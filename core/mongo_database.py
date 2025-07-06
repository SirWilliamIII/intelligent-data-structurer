"""
MongoDB database manager for intelligent data processing.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from loguru import logger

from .config import settings


class MongoDBManager:
    """Manages MongoDB connections and collections."""
    
    def __init__(self, database_name: str = "intelligent_data"):
        self.database_name = database_name
        self.client = None
        self.database = None
        self._collections = {}
        
    async def connect(self):
        """Connect to MongoDB."""
        try:
            # Default MongoDB connection string
            mongo_url = getattr(settings, 'mongo_url', 'mongodb://localhost:27017')
            
            self.client = AsyncIOMotorClient(mongo_url)
            self.database = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {self.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def get_collection(self, collection_name: str):
        """Get a collection, creating indexes if needed."""
        if collection_name not in self._collections:
            collection = self.database[collection_name]
            
            # Create indexes for better performance
            indexes = [
                IndexModel([("priority", DESCENDING)]),
                IndexModel([("domain", ASCENDING)]),
                IndexModel([("content_type", ASCENDING)]),
                IndexModel([("extracted_at", DESCENDING)]),
                IndexModel([("source_file", ASCENDING)]),
                IndexModel([("content", TEXT)]),  # Full-text search
            ]
            
            try:
                await collection.create_indexes(indexes)
                logger.info(f"Created indexes for collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to create indexes for {collection_name}: {e}")
            
            self._collections[collection_name] = collection
        
        return self._collections[collection_name]
    
    async def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert a document into the specified collection."""
        try:
            collection = await self.get_collection(collection_name)
            
            # Add metadata
            document['inserted_at'] = datetime.utcnow()
            
            # Handle sets and other non-serializable types
            processed_doc = self._process_document(document)
            
            result = await collection.insert_one(processed_doc)
            doc_id = str(result.inserted_id)
            
            logger.info(f"Inserted document into {collection_name}, ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to insert document into {collection_name}: {e}")
            raise
    
    async def find_documents(self, collection_name: str, query: Dict[str, Any] = None, 
                           limit: int = 100, sort: List[tuple] = None) -> List[Dict]:
        """Find documents in the specified collection."""
        try:
            collection = await self.get_collection(collection_name)
            query = query or {}
            
            cursor = collection.find(query)
            
            if sort:
                cursor = cursor.sort(sort)
            else:
                cursor = cursor.sort([("extracted_at", DESCENDING)])
            
            cursor = cursor.limit(limit)
            
            documents = []
            async for doc in cursor:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to find documents in {collection_name}: {e}")
            return []
    
    async def find_one_document(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict]:
        """Find a single document in the specified collection."""
        try:
            collection = await self.get_collection(collection_name)
            doc = await collection.find_one(query)
            
            if doc:
                doc['_id'] = str(doc['_id'])
                return doc
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find document in {collection_name}: {e}")
            return None
    
    async def update_document(self, collection_name: str, query: Dict[str, Any], 
                            update: Dict[str, Any]) -> bool:
        """Update a document in the specified collection."""
        try:
            collection = await self.get_collection(collection_name)
            
            # Add update metadata
            update['$set'] = update.get('$set', {})
            update['$set']['updated_at'] = datetime.utcnow()
            
            result = await collection.update_one(query, update)
            
            if result.modified_count > 0:
                logger.info(f"Updated document in {collection_name}")
                return True
            else:
                logger.warning(f"No document updated in {collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update document in {collection_name}: {e}")
            return False
    
    async def delete_document(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """Delete a document from the specified collection."""
        try:
            collection = await self.get_collection(collection_name)
            result = await collection.delete_one(query)
            
            if result.deleted_count > 0:
                logger.info(f"Deleted document from {collection_name}")
                return True
            else:
                logger.warning(f"No document deleted from {collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document from {collection_name}: {e}")
            return False
    
    async def count_documents(self, collection_name: str, query: Dict[str, Any] = None) -> int:
        """Count documents in the specified collection."""
        try:
            collection = await self.get_collection(collection_name)
            query = query or {}
            return await collection.count_documents(query)
            
        except Exception as e:
            logger.error(f"Failed to count documents in {collection_name}: {e}")
            return 0
    
    async def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            collections = await self.database.list_collection_names()
            return [col for col in collections if not col.startswith('system.')]
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, int]:
        """Get document counts for all collections."""
        collections = await self.list_collections()
        stats = {}
        
        for collection_name in collections:
            try:
                count = await self.count_documents(collection_name)
                stats[collection_name] = count
            except Exception as e:
                logger.error(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = 0
        
        return stats
    
    async def create_text_index(self, collection_name: str, fields: List[str]):
        """Create a text index on specified fields for full-text search."""
        try:
            collection = await self.get_collection(collection_name)
            
            # Create text index
            index_spec = [(field, TEXT) for field in fields]
            await collection.create_index(index_spec)
            
            logger.info(f"Created text index on {fields} for {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create text index for {collection_name}: {e}")
    
    async def search_text(self, collection_name: str, search_term: str, 
                         limit: int = 50) -> List[Dict]:
        """Perform full-text search on a collection."""
        try:
            collection = await self.get_collection(collection_name)
            
            cursor = collection.find(
                {"$text": {"$search": search_term}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            documents = []
            async for doc in cursor:
                doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search text in {collection_name}: {e}")
            return []
    
    def _process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Process document to handle non-serializable types."""
        processed = {}
        
        for key, value in document.items():
            if isinstance(value, set):
                processed[key] = list(value)
            elif isinstance(value, dict):
                processed[key] = self._process_document(value)
            elif isinstance(value, list):
                processed[key] = [
                    self._process_document(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                processed[key] = value
        
        return processed
    
    async def aggregate(self, collection_name: str, pipeline: List[Dict]) -> List[Dict]:
        """Run aggregation pipeline on a collection."""
        try:
            collection = await self.get_collection(collection_name)
            
            cursor = collection.aggregate(pipeline)
            results = []
            
            async for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run aggregation on {collection_name}: {e}")
            return []
    
    async def bulk_insert(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents in bulk."""
        try:
            collection = await self.get_collection(collection_name)
            
            # Process all documents
            processed_docs = []
            for doc in documents:
                doc['inserted_at'] = datetime.utcnow()
                processed_docs.append(self._process_document(doc))
            
            result = await collection.insert_many(processed_docs)
            doc_ids = [str(obj_id) for obj_id in result.inserted_ids]
            
            logger.info(f"Bulk inserted {len(doc_ids)} documents into {collection_name}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to bulk insert documents into {collection_name}: {e}")
            raise


# Global instance
mongo_manager = MongoDBManager()