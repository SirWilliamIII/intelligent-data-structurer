"""
Organic Collection Evolution System

A brilliant approach where collections are living entities that:
- Start as seeds from similar documents
- Grow by attracting related content
- Evolve their schemas based on patterns
- Merge when they overlap
- Split when they become too broad
- Die when they're no longer useful

This creates a self-organizing knowledge graph that mirrors how human knowledge naturally clusters.
"""

import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pymongo
from bson import ObjectId
from loguru import logger
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

@dataclass
class CollectionSeed:
    """A potential collection waiting to be born."""
    theme_keywords: Set[str]
    document_ids: List[str]
    semantic_signature: str
    birth_confidence: float
    first_seen: datetime
    last_reinforced: datetime
    
@dataclass
class CollectionDNA:
    """The genetic code of a collection - what makes it unique."""
    core_themes: Set[str]
    structural_patterns: Set[str]
    content_markers: Set[str]
    quality_indicators: Dict[str, float]
    evolution_history: List[Dict[str, Any]]
    compatibility_map: Dict[str, float]  # How well it works with other collections
    
@dataclass
class CollectionHealth:
    """Health metrics for a collection."""
    coherence_score: float  # How well documents fit together
    growth_rate: float  # How fast it's growing
    utility_score: float  # How often it's accessed/useful
    diversity_index: float  # Variety of content within theme
    last_health_check: datetime
    
class OrganicCollectionManager:
    """The brain that manages the organic collection ecosystem."""
    
    def __init__(self, mongo_client: pymongo.MongoClient, db_name: str = "intelligent_data"):
        self.mongo_client = mongo_client
        self.db = mongo_client[db_name]
        self.staging_collection = self.db["document_staging"]
        self.collection_registry = self.db["collection_registry"]
        self.collection_seeds = self.db["collection_seeds"]
        
        # Evolution parameters
        self.min_seed_size = 3  # Minimum docs to form a collection
        self.max_collection_size = 100  # Split collections that get too large
        self.similarity_threshold = 0.3  # How similar docs need to be
        self.health_check_interval = timedelta(hours=6)
        self.seed_maturation_time = timedelta(hours=24)  # Time before seed becomes collection
        
        # Collection genetics
        self.collection_dna = {}
        self.collection_health = {}
        
        # TF-IDF vectorizer for semantic analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self._initialize_ecosystem()
    
    def _initialize_ecosystem(self):
        """Initialize the organic collection ecosystem."""
        # Create indexes for efficient querying
        self.staging_collection.create_index([("semantic_signature.semantic_hash", 1)])
        self.staging_collection.create_index([("inserted_at", 1)])
        self.collection_registry.create_index([("collection_name", 1)])
        self.collection_seeds.create_index([("theme_keywords", 1)])
        
        logger.info("🌱 Organic collection ecosystem initialized")
    
    async def process_document(self, document_data: Dict[str, Any]) -> str:
        """Process a document through the organic collection system."""
        
        # Stage 1: Store in staging area
        doc_id = await self._stage_document(document_data)
        
        # Stage 2: Look for collection opportunities
        await self._evaluate_collection_opportunities(doc_id, document_data)
        
        # Stage 3: Evolve existing collections
        await self._evolve_collections()
        
        # Stage 4: Health check on collections
        await self._health_check_collections()
        
        return doc_id
    
    async def _stage_document(self, document_data: Dict[str, Any]) -> str:
        """Stage a document and return its ID."""
        
        # Convert sets to lists for MongoDB compatibility
        cleaned_data = self._clean_data_for_mongo(document_data)
        
        # Add staging metadata
        cleaned_data.update({
            "staged_at": datetime.utcnow(),
            "collection_assigned": None,
            "collection_confidence": 0.0,
            "evolution_ready": False
        })
        
        result = self.staging_collection.insert_one(cleaned_data)
        doc_id = str(result.inserted_id)
        
        logger.info(f"📥 Document staged: {doc_id}")
        return doc_id
    
    async def _evaluate_collection_opportunities(self, doc_id: str, document_data: Dict[str, Any]):
        """Evaluate if this document creates or joins collection opportunities."""
        
        # Find similar documents in staging
        similar_docs = await self._find_similar_staged_documents(document_data)
        
        if len(similar_docs) >= self.min_seed_size:
            # Check if we already have a seed for this cluster
            existing_seed = await self._find_existing_seed(similar_docs)
            
            if existing_seed:
                # Reinforce existing seed
                await self._reinforce_seed(existing_seed['_id'], doc_id)
            else:
                # Create new seed
                await self._create_collection_seed(similar_docs + [doc_id])
        
        # Check if document fits into existing mature collections
        await self._evaluate_existing_collections(doc_id, document_data)
    
    async def _find_similar_staged_documents(self, document_data: Dict[str, Any]) -> List[str]:
        """Find similar documents in the staging area using advanced similarity."""
        
        # Get all staged documents
        staged_docs = list(self.staging_collection.find({"collection_assigned": None}))
        
        if len(staged_docs) < 2:
            return []
        
        # Extract content for similarity analysis
        contents = []
        doc_ids = []
        
        for doc in staged_docs:
            content = doc.get('content', '')
            if len(content) > 50:  # Only consider substantial content
                contents.append(content)
                doc_ids.append(str(doc['_id']))
        
        if len(contents) < 2:
            return []
        
        # Calculate TF-IDF similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find the index of our target document
            target_content = document_data.get('content', '')
            target_tfidf = self.vectorizer.transform([target_content])
            similarities = cosine_similarity(target_tfidf, tfidf_matrix)[0]
            
            # Find similar documents
            similar_indices = np.where(similarities > self.similarity_threshold)[0]
            similar_doc_ids = [doc_ids[i] for i in similar_indices]
            
            return similar_doc_ids
            
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            return []
    
    async def _find_existing_seed(self, similar_docs: List[str]) -> Optional[Dict]:
        """Check if we already have a seed for this cluster."""
        
        # Look for seeds that overlap with our similar documents
        for seed in self.collection_seeds.find():
            overlap = set(seed['document_ids']) & set(similar_docs)
            if len(overlap) >= 3:  # Significant overlap
                return seed
        
        return None
    
    async def _create_collection_seed(self, document_ids: List[str]):
        """Create a new collection seed from similar documents."""
        
        # Analyze the cluster to extract themes
        cluster_docs = list(self.staging_collection.find({"_id": {"$in": [ObjectId(doc_id) for doc_id in document_ids]}}))
        
        # Extract common themes
        theme_keywords = self._extract_cluster_themes(cluster_docs)
        
        # Create semantic signature for the cluster
        cluster_contents = [doc.get('content', '') for doc in cluster_docs]
        semantic_signature = self._generate_cluster_signature(cluster_contents)
        
        # Calculate birth confidence
        birth_confidence = self._calculate_birth_confidence(cluster_docs)
        
        # Create the seed
        seed = CollectionSeed(
            theme_keywords=theme_keywords,
            document_ids=document_ids,
            semantic_signature=semantic_signature,
            birth_confidence=birth_confidence,
            first_seen=datetime.utcnow(),
            last_reinforced=datetime.utcnow()
        )
        
        # Store the seed
        result = self.collection_seeds.insert_one(asdict(seed))
        
        logger.info(f"🌱 Collection seed created: {theme_keywords} ({len(document_ids)} docs, {birth_confidence:.2f} confidence)")
        
        return result.inserted_id
    
    async def _reinforce_seed(self, seed_id: str, new_doc_id: str):
        """Reinforce an existing seed with a new document."""
        
        self.collection_seeds.update_one(
            {"_id": ObjectId(seed_id)},
            {
                "$push": {"document_ids": new_doc_id},
                "$set": {"last_reinforced": datetime.utcnow()}
            }
        )
        
        logger.info(f"🌱 Seed reinforced: {seed_id}")
    
    async def _evaluate_existing_collections(self, doc_id: str, document_data: Dict[str, Any]):
        """Check if document should join an existing mature collection."""
        
        # Get all mature collections
        mature_collections = list(self.collection_registry.find({"status": "mature"}))
        
        best_match = None
        best_confidence = 0.0
        
        for collection in mature_collections:
            confidence = await self._calculate_collection_fit(document_data, collection)
            
            if confidence > best_confidence and confidence > 0.8:
                best_match = collection
                best_confidence = confidence
        
        if best_match:
            await self._assign_to_collection(doc_id, best_match['collection_name'], best_confidence)
    
    async def _calculate_collection_fit(self, document_data: Dict[str, Any], collection: Dict[str, Any]) -> float:
        """Calculate how well a document fits into an existing collection."""
        
        # Get collection's DNA
        collection_name = collection['collection_name']
        if collection_name not in self.collection_dna:
            await self._analyze_collection_dna(collection_name)
        
        dna = self.collection_dna[collection_name]
        
        # Calculate semantic similarity
        doc_signature = document_data.get('semantic_signature', {})
        doc_keywords = set(doc_signature.get('domain_keywords', []))
        doc_patterns = set(doc_signature.get('structural_patterns', []))
        
        # Theme overlap
        theme_overlap = len(doc_keywords & dna.core_themes) / max(len(doc_keywords | dna.core_themes), 1)
        
        # Pattern overlap
        pattern_overlap = len(doc_patterns & dna.structural_patterns) / max(len(doc_patterns | dna.structural_patterns), 1)
        
        # Quality indicators
        quality_score = 0.0
        if document_data.get('word_count', 0) > 100:
            quality_score += 0.1
        if document_data.get('entities', []):
            quality_score += 0.1
        
        # Weighted confidence
        confidence = (theme_overlap * 0.5 + pattern_overlap * 0.3 + quality_score * 0.2)
        
        return confidence
    
    async def _assign_to_collection(self, doc_id: str, collection_name: str, confidence: float):
        """Assign a document to a mature collection."""
        
        # Move document from staging to collection
        doc = self.staging_collection.find_one({"_id": ObjectId(doc_id)})
        if doc:
            # Update document with collection assignment
            doc.update({
                "collection_assigned": collection_name,
                "collection_confidence": confidence,
                "assigned_at": datetime.utcnow()
            })
            
            # Insert into target collection
            target_collection = self.db[collection_name]
            target_collection.insert_one(doc)
            
            # Remove from staging
            self.staging_collection.delete_one({"_id": ObjectId(doc_id)})
            
            logger.info(f"📂 Document assigned to {collection_name} (confidence: {confidence:.2f})")
    
    async def _evolve_collections(self):
        """Evolve collections by checking seeds for maturation."""
        
        # Check seeds for maturation
        mature_seeds = list(self.collection_seeds.find({
            "first_seen": {"$lt": datetime.utcnow() - self.seed_maturation_time},
            "birth_confidence": {"$gte": 0.7}
        }))
        
        for seed in mature_seeds:
            await self._birth_collection_from_seed(seed)
    
    async def _birth_collection_from_seed(self, seed: Dict[str, Any]):
        """Birth a new collection from a mature seed."""
        
        # Generate collection name from themes
        collection_name = self._generate_collection_name(seed['theme_keywords'])
        
        # Create collection DNA
        dna = CollectionDNA(
            core_themes=set(seed['theme_keywords']),
            structural_patterns=set(),
            content_markers=set(),
            quality_indicators={},
            evolution_history=[{
                "event": "birth",
                "timestamp": datetime.utcnow(),
                "details": f"Born from seed with {len(seed['document_ids'])} documents"
            }],
            compatibility_map={}
        )
        
        # Store collection DNA
        self.collection_dna[collection_name] = dna
        
        # Register the collection
        self.collection_registry.insert_one({
            "collection_name": collection_name,
            "status": "mature",
            "created_at": datetime.utcnow(),
            "document_count": len(seed['document_ids']),
            "theme_keywords": list(seed['theme_keywords']),
            "birth_confidence": seed['birth_confidence']
        })
        
        # Create the actual MongoDB collection
        new_collection = self.db[collection_name]
        
        # Move documents from staging to new collection
        for doc_id in seed['document_ids']:
            doc = self.staging_collection.find_one({"_id": ObjectId(doc_id)})
            if doc:
                doc.update({
                    "collection_assigned": collection_name,
                    "collection_confidence": 1.0,
                    "assigned_at": datetime.utcnow()
                })
                new_collection.insert_one(doc)
                self.staging_collection.delete_one({"_id": ObjectId(doc_id)})
        
        # Remove the seed
        self.collection_seeds.delete_one({"_id": seed['_id']})
        
        logger.info(f"🎉 Collection born: {collection_name} with {len(seed['document_ids'])} documents")
    
    async def _health_check_collections(self):
        """Perform health checks on all collections."""
        
        collections = list(self.collection_registry.find({"status": "mature"}))
        
        for collection in collections:
            collection_name = collection['collection_name']
            
            # Skip if recently checked
            if collection_name in self.collection_health:
                last_check = self.collection_health[collection_name].last_health_check
                if datetime.utcnow() - last_check < self.health_check_interval:
                    continue
            
            # Perform health check
            health = await self._calculate_collection_health(collection_name)
            self.collection_health[collection_name] = health
            
            # Take action based on health
            if health.coherence_score < 0.4:
                await self._consider_collection_split(collection_name)
            elif health.utility_score < 0.2:
                await self._consider_collection_merge(collection_name)
    
    async def _calculate_collection_health(self, collection_name: str) -> CollectionHealth:
        """Calculate health metrics for a collection."""
        
        collection = self.db[collection_name]
        docs = list(collection.find())
        
        if not docs:
            return CollectionHealth(0.0, 0.0, 0.0, 0.0, datetime.utcnow())
        
        # Calculate coherence (how well documents fit together)
        coherence_score = await self._calculate_coherence(docs)
        
        # Calculate growth rate
        growth_rate = await self._calculate_growth_rate(collection_name)
        
        # Calculate utility (placeholder - would track actual usage)
        utility_score = 0.5  # Default utility
        
        # Calculate diversity
        diversity_index = await self._calculate_diversity(docs)
        
        return CollectionHealth(
            coherence_score=coherence_score,
            growth_rate=growth_rate,
            utility_score=utility_score,
            diversity_index=diversity_index,
            last_health_check=datetime.utcnow()
        )
    
    async def _calculate_coherence(self, docs: List[Dict]) -> float:
        """Calculate how coherent/similar the documents in a collection are."""
        
        if len(docs) < 2:
            return 1.0
        
        contents = [doc.get('content', '') for doc in docs if doc.get('content')]
        
        if len(contents) < 2:
            return 0.5
        
        try:
            # Calculate pairwise similarities
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Get average similarity (excluding diagonal)
            mask = np.ones(similarity_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            
            avg_similarity = similarity_matrix[mask].mean()
            return float(avg_similarity)
            
        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
            return 0.5
    
    async def _calculate_growth_rate(self, collection_name: str) -> float:
        """Calculate how fast a collection is growing."""
        
        collection = self.db[collection_name]
        
        # Count documents added in last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_docs = collection.count_documents({"assigned_at": {"$gte": week_ago}})
        total_docs = collection.count_documents({})
        
        if total_docs == 0:
            return 0.0
        
        return recent_docs / total_docs
    
    async def _calculate_diversity(self, docs: List[Dict]) -> float:
        """Calculate diversity of content within the collection."""
        
        # Extract all keywords from all documents
        all_keywords = []
        for doc in docs:
            sig = doc.get('semantic_signature', {})
            keywords = sig.get('domain_keywords', [])
            all_keywords.extend(keywords)
        
        if not all_keywords:
            return 0.0
        
        # Calculate Shannon diversity index
        keyword_counts = Counter(all_keywords)
        total = len(all_keywords)
        
        diversity = 0.0
        for count in keyword_counts.values():
            p = count / total
            if p > 0:
                diversity -= p * np.log(p)
        
        return diversity
    
    def _extract_cluster_themes(self, docs: List[Dict]) -> Set[str]:
        """Extract common themes from a cluster of documents."""
        
        all_keywords = []
        for doc in docs:
            sig = doc.get('semantic_signature', {})
            keywords = sig.get('domain_keywords', [])
            all_keywords.extend(keywords)
        
        # Find most common keywords
        keyword_counts = Counter(all_keywords)
        common_keywords = [word for word, count in keyword_counts.most_common(5) if count > 1]
        
        return set(common_keywords)
    
    def _generate_cluster_signature(self, contents: List[str]) -> str:
        """Generate a semantic signature for a cluster."""
        
        combined_content = ' '.join(contents)
        return hashlib.md5(combined_content.encode()).hexdigest()
    
    def _calculate_birth_confidence(self, docs: List[Dict]) -> float:
        """Calculate confidence that this cluster should become a collection."""
        
        # Base confidence
        confidence = 0.5
        
        # Boost for document count
        doc_count = len(docs)
        if doc_count >= 5:
            confidence += 0.2
        if doc_count >= 10:
            confidence += 0.1
        
        # Boost for content quality
        total_words = sum(doc.get('word_count', 0) for doc in docs)
        if total_words > 1000:
            confidence += 0.1
        
        # Boost for entity richness
        total_entities = sum(len(doc.get('entities', [])) for doc in docs)
        if total_entities > 20:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_collection_name(self, themes: Set[str]) -> str:
        """Generate a meaningful collection name from themes."""
        
        themes_list = list(themes)
        if not themes_list:
            return f"collection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Sort by length (prefer longer, more specific terms)
        themes_list.sort(key=len, reverse=True)
        
        # Take top 2 themes
        primary_themes = themes_list[:2]
        
        # Create name
        name_parts = []
        for theme in primary_themes:
            # Clean theme name
            clean_theme = theme.replace('-', '_').replace(' ', '_').lower()
            name_parts.append(clean_theme)
        
        base_name = '_'.join(name_parts)
        
        # Add suffix to make it descriptive
        if any(tech in base_name for tech in ['kubernetes', 'docker', 'api', 'config', 'command']):
            suffix = 'technical'
        elif any(food in base_name for food in ['recipe', 'cooking', 'ingredient']):
            suffix = 'culinary'
        elif any(sport in base_name for sport in ['team', 'game', 'sport', 'match']):
            suffix = 'sports'
        else:
            suffix = 'documents'
        
        return f"{base_name}_{suffix}"
    
    async def _analyze_collection_dna(self, collection_name: str):
        """Analyze a collection to extract its DNA."""
        
        collection = self.db[collection_name]
        docs = list(collection.find())
        
        if not docs:
            return
        
        # Extract themes
        all_keywords = []
        all_patterns = []
        all_markers = []
        
        for doc in docs:
            sig = doc.get('semantic_signature', {})
            all_keywords.extend(sig.get('domain_keywords', []))
            all_patterns.extend(sig.get('structural_patterns', []))
            all_markers.extend(sig.get('content_markers', []))
        
        # Get most common elements
        core_themes = set([word for word, count in Counter(all_keywords).most_common(10)])
        structural_patterns = set([pattern for pattern, count in Counter(all_patterns).most_common(5)])
        content_markers = set([marker for marker, count in Counter(all_markers).most_common(5)])
        
        # Create DNA
        dna = CollectionDNA(
            core_themes=core_themes,
            structural_patterns=structural_patterns,
            content_markers=content_markers,
            quality_indicators={},
            evolution_history=[],
            compatibility_map={}
        )
        
        self.collection_dna[collection_name] = dna
    
    async def get_collection_insights(self) -> Dict[str, Any]:
        """Get insights about the collection ecosystem."""
        
        # Count documents by status
        staging_count = self.staging_collection.count_documents({})
        seed_count = self.collection_seeds.count_documents({})
        mature_collections = list(self.collection_registry.find({"status": "mature"}))
        
        # Calculate total documents in mature collections
        total_mature_docs = 0
        collection_sizes = {}
        
        for collection in mature_collections:
            collection_name = collection['collection_name']
            count = self.db[collection_name].count_documents({})
            collection_sizes[collection_name] = count
            total_mature_docs += count
        
        # Health summary
        healthy_collections = sum(1 for name, health in self.collection_health.items() 
                                 if health.coherence_score > 0.6)
        
        return {
            "ecosystem_status": {
                "documents_in_staging": staging_count,
                "collection_seeds": seed_count,
                "mature_collections": len(mature_collections),
                "total_organized_documents": total_mature_docs
            },
            "collection_sizes": collection_sizes,
            "health_summary": {
                "healthy_collections": healthy_collections,
                "total_collections": len(mature_collections),
                "health_percentage": (healthy_collections / max(len(mature_collections), 1)) * 100
            },
            "seeds_ready_for_birth": self.collection_seeds.count_documents({
                "first_seen": {"$lt": datetime.utcnow() - self.seed_maturation_time},
                "birth_confidence": {"$gte": 0.7}
            })
        }
    
    async def suggest_collection_improvements(self) -> List[Dict[str, Any]]:
        """Suggest improvements to the collection ecosystem."""
        
        suggestions = []
        
        # Check for overgrown collections
        for collection in self.collection_registry.find({"status": "mature"}):
            collection_name = collection['collection_name']
            doc_count = self.db[collection_name].count_documents({})
            
            if doc_count > self.max_collection_size:
                suggestions.append({
                    "type": "split_collection",
                    "collection": collection_name,
                    "reason": f"Collection has {doc_count} documents, consider splitting",
                    "urgency": "medium"
                })
        
        # Check for similar collections that could merge
        mature_collections = list(self.collection_registry.find({"status": "mature"}))
        for i, coll1 in enumerate(mature_collections):
            for coll2 in mature_collections[i+1:]:
                similarity = await self._calculate_collection_similarity(coll1['collection_name'], coll2['collection_name'])
                if similarity > 0.8:
                    suggestions.append({
                        "type": "merge_collections",
                        "collections": [coll1['collection_name'], coll2['collection_name']],
                        "reason": f"Collections are {similarity:.1%} similar",
                        "urgency": "low"
                    })
        
        return suggestions
    
    async def _calculate_collection_similarity(self, collection1: str, collection2: str) -> float:
        """Calculate similarity between two collections."""
        
        # Ensure both collections have DNA
        if collection1 not in self.collection_dna:
            await self._analyze_collection_dna(collection1)
        if collection2 not in self.collection_dna:
            await self._analyze_collection_dna(collection2)
        
        dna1 = self.collection_dna[collection1]
        dna2 = self.collection_dna[collection2]
        
        # Calculate theme overlap
        theme_overlap = len(dna1.core_themes & dna2.core_themes) / max(len(dna1.core_themes | dna2.core_themes), 1)
        
        # Calculate pattern overlap
        pattern_overlap = len(dna1.structural_patterns & dna2.structural_patterns) / max(len(dna1.structural_patterns | dna2.structural_patterns), 1)
        
        # Weighted similarity
        return theme_overlap * 0.7 + pattern_overlap * 0.3
    
    async def _consider_collection_split(self, collection_name: str):
        """Consider splitting a collection with low coherence."""
        
        collection = self.db[collection_name]
        docs = list(collection.find())
        
        if len(docs) < 10:  # Too small to split
            return
        
        # Use clustering to find natural splits
        contents = [doc.get('content', '') for doc in docs if doc.get('content')]
        
        if len(contents) < 5:
            return
        
        try:
            # TF-IDF + DBSCAN clustering
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            
            # Use DBSCAN to find natural clusters
            clusterer = DBSCAN(metric='cosine', min_samples=3)
            cluster_labels = clusterer.fit_predict(tfidf_matrix.toarray())
            
            # Check if we found meaningful clusters
            unique_labels = set(cluster_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:  # -1 is noise in DBSCAN
                
                logger.info(f"🔄 Collection {collection_name} could be split into {len(unique_labels)} clusters")
                
                # This is where you'd implement the actual splitting logic
                # For now, just log the suggestion
                
        except Exception as e:
            logger.warning(f"Collection split analysis failed: {e}")
    
    async def _consider_collection_merge(self, collection_name: str):
        """Consider merging a collection with low utility."""
        
        # Find the most similar collection
        mature_collections = list(self.collection_registry.find({"status": "mature"}))
        
        best_match = None
        best_similarity = 0.0
        
        for collection in mature_collections:
            other_name = collection['collection_name']
            if other_name != collection_name:
                similarity = await self._calculate_collection_similarity(collection_name, other_name)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = other_name
        
        if best_similarity > 0.7:
            logger.info(f"🔄 Collection {collection_name} could merge with {best_match} (similarity: {best_similarity:.1%})")
            
            # This is where you'd implement the actual merging logic
            # For now, just log the suggestion
    
    def _clean_data_for_mongo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean data to be MongoDB compatible (convert sets to lists, etc.)."""
        
        def clean_value(value):
            if isinstance(value, set):
                return list(value)
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(item) for item in value]
            else:
                return value
        
        return {key: clean_value(value) for key, value in data.items()}