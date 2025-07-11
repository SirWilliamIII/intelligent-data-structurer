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
from core.config import settings

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
        
        # Evolution parameters - now configurable
        self.staging_threshold = settings.staging_threshold  # Documents needed before creating collection
        self.min_seed_size = settings.min_seed_size  # Minimum docs to form a collection seed
        self.max_collection_size = 100  # Split collections that get too large
        self.similarity_threshold = 0.15  # How similar docs need to be
        self.health_check_interval = timedelta(hours=6)
        self.seed_maturation_time = timedelta(minutes=1)  # Time before seed becomes collection
        self.max_staging_time = timedelta(hours=settings.max_staging_time_hours)  # Max time in staging
        self.collection_birth_confidence = settings.collection_birth_confidence  # Confidence threshold for new collections
        
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
        
        # Stage 4: Monitor mature collections for subdivision opportunities
        await self._monitor_collection_subdivision()
        
        # Stage 5: Health check on collections
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
        
        # First check if we've reached the staging threshold
        staging_count = self.staging_collection.count_documents({"collection_assigned": None})
        
        if staging_count < self.staging_threshold:
            # Not enough documents in staging yet - just log and return
            logger.info(f"📥 Document staged ({staging_count}/{self.staging_threshold} threshold). Waiting for more documents.")
            return
        
        # We've reached the threshold - now evaluate clustering opportunities
        logger.info(f"⚙️ Staging threshold reached ({staging_count} documents). Evaluating clustering opportunities.")
        
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
        """Find similar documents in the staging area using semantic signature similarity."""
        
        # Get all staged documents
        staged_docs = list(self.staging_collection.find({"collection_assigned": None}))
        
        if len(staged_docs) < 2:
            return []
        
        # Get target document's semantic signature
        target_signature = document_data.get('semantic_signature', {})
        target_keywords = set(target_signature.get('domain_keywords', []))
        target_patterns = set(target_signature.get('structural_patterns', []))
        target_markers = set(target_signature.get('content_markers', []))
        
        if not target_keywords and not target_patterns:
            logger.warning("Target document has no semantic signature - falling back to content similarity")
            return await self._fallback_content_similarity(document_data, staged_docs)
        
        similar_docs = []
        
        for doc in staged_docs:
            doc_signature = doc.get('semantic_signature', {})
            doc_keywords = set(doc_signature.get('domain_keywords', []))
            doc_patterns = set(doc_signature.get('structural_patterns', []))
            doc_markers = set(doc_signature.get('content_markers', []))
            
            # Calculate semantic similarity using Jaccard similarity
            keyword_similarity = self._jaccard_similarity(target_keywords, doc_keywords)
            pattern_similarity = self._jaccard_similarity(target_patterns, doc_patterns)
            marker_similarity = self._jaccard_similarity(target_markers, doc_markers)
            
            # Weighted similarity score (keywords are most important)
            similarity_score = (
                keyword_similarity * 0.6 +
                pattern_similarity * 0.3 +
                marker_similarity * 0.1
            )
            
            # Check for domain-specific high-value keywords
            domain_boost = self._calculate_domain_boost(target_keywords, doc_keywords)
            similarity_score += domain_boost
            
            logger.debug(f"Semantic similarity: {similarity_score:.3f} "
                        f"(kw:{keyword_similarity:.3f}, pat:{pattern_similarity:.3f}, "
                        f"mark:{marker_similarity:.3f}, boost:{domain_boost:.3f})")
            
            # Use higher threshold for semantic similarity (more accurate)
            if similarity_score > 0.7:  # Raised from 0.5 for accuracy
                similar_docs.append(str(doc['_id']))
                logger.info(f"Found semantically similar document: {doc.get('source_file', 'unknown')} "
                           f"(similarity: {similarity_score:.3f})")
        
        return similar_docs
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _calculate_domain_boost(self, target_keywords: set, doc_keywords: set) -> float:
        """Calculate additional similarity boost for domain-specific keyword matches."""
        
        # High-value domain keywords that strongly indicate similarity
        domain_clusters = {
            'sports': {'nfl', 'mlb', 'football', 'baseball', 'team', 'league', 'sports', 'game'},
            'medical': {'medical', 'patient', 'doctor', 'hospital', 'diagnosis', 'treatment'},
            'technical': {'kubernetes', 'docker', 'git', 'api', 'software', 'code', 'system'},
            'financial': {'expense', 'invoice', 'payment', 'financial', 'cost', 'budget'},
            'business': {'meeting', 'employee', 'contract', 'business', 'company', 'organization'}
        }
        
        boost = 0.0
        
        for domain, domain_keywords in domain_clusters.items():
            target_domain_match = len(target_keywords & domain_keywords)
            doc_domain_match = len(doc_keywords & domain_keywords)
            
            # Both documents have keywords from the same domain
            if target_domain_match >= 2 and doc_domain_match >= 2:
                # Strong domain match - significant boost
                shared_domain_keywords = len((target_keywords & domain_keywords) & (doc_keywords & domain_keywords))
                if shared_domain_keywords >= 2:
                    boost += 0.3
                    logger.debug(f"Strong {domain} domain match: +0.3 boost")
                elif shared_domain_keywords >= 1:
                    boost += 0.15
                    logger.debug(f"Moderate {domain} domain match: +0.15 boost")
        
        return min(boost, 0.4)  # Cap boost at 0.4
    
    async def _fallback_content_similarity(self, document_data: Dict[str, Any], staged_docs: List[Dict]) -> List[str]:
        """Fallback to content similarity when semantic signatures are unavailable."""
        
        contents = []
        doc_ids = []
        
        for doc in staged_docs:
            content = doc.get('content', '')
            if len(content) > 50:
                contents.append(content)
                doc_ids.append(str(doc['_id']))
        
        if len(contents) < 2:
            return []
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(contents)
            target_content = document_data.get('content', '')
            target_tfidf = self.vectorizer.transform([target_content])
            similarities = cosine_similarity(target_tfidf, tfidf_matrix)[0]
            
            # Use higher threshold for content similarity too
            similar_indices = np.where(similarities > 0.8)[0]  # Raised from 0.5
            similar_doc_ids = [doc_ids[i] for i in similar_indices]
            
            return similar_doc_ids
            
        except Exception as e:
            logger.warning(f"Content similarity calculation failed: {e}")
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
        
        # Store the seed (convert sets to lists for MongoDB)
        seed_data = self._clean_data_for_mongo(asdict(seed))
        result = self.collection_seeds.insert_one(seed_data)
        
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
            "birth_confidence": {"$gte": self.collection_birth_confidence}
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
        """Extract common themes from a cluster of documents, prioritizing business document types."""
        
        # First, check for business document types (highest priority)
        business_types = []
        business_categories = []
        
        for doc in docs:
            content_type = doc.get('content_type', '')
            business_category = doc.get('business_category', '')
            
            # Map business document types to clean theme names
            if content_type:
                business_types.append(content_type)
            if business_category:
                business_categories.append(business_category)
        
        # If we have consistent business document types, use them
        type_counts = Counter(business_types)
        category_counts = Counter(business_categories)
        
        # Check if majority of documents have the same business type
        if type_counts and len(docs) > 1:
            most_common_type, type_count = type_counts.most_common(1)[0]
            if type_count >= len(docs) * 0.6:  # 60% threshold
                # Use business type as primary theme
                themes = {most_common_type}
                
                # Add category if consistent
                if category_counts:
                    most_common_category, category_count = category_counts.most_common(1)[0]
                    if category_count >= len(docs) * 0.6:
                        themes.add(most_common_category.lower().replace(' ', '_'))
                
                return themes
        
        # Fallback to original semantic keyword extraction
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
        """Generate a meaningful collection name from themes, prioritizing business document types."""
        
        themes_list = list(themes)
        if not themes_list:
            return f"collection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Check for business document types first
        business_document_types = {
            'invoice', 'expense_report', 'purchase_order', 'receipt', 'financial_statement',
            'meeting_minutes', 'business_email', 'memo', 'proposal', 'report',
            'resume', 'job_posting', 'timesheet', 'performance_review',
            'contract', 'nda', 'work_order', 'project_plan', 'sales_quote'
        }
        
        business_categories = {'financial', 'communication', 'human_resources', 'legal', 'operations', 'sales'}
        
        # Look for business document types in themes
        business_types = [theme for theme in themes_list if theme in business_document_types]
        categories = [theme for theme in themes_list if theme in business_categories]
        
        if business_types:
            # Use business document type as primary name
            primary_type = business_types[0]
            
            # Map to proper collection names
            if primary_type == 'expense_report':
                return 'expense_reports'
            elif primary_type == 'invoice':
                return 'invoices'
            elif primary_type == 'meeting_minutes':
                return 'meeting_minutes'
            elif primary_type == 'business_email':
                return 'business_emails'
            elif primary_type == 'purchase_order':
                return 'purchase_orders'
            elif primary_type == 'contract':
                return 'contracts'
            elif primary_type == 'project_plan':
                return 'project_plans'
            elif primary_type == 'sales_quote':
                return 'sales_quotes'
            elif primary_type == 'work_order':
                return 'work_orders'
            elif primary_type == 'timesheet':
                return 'timesheets'
            elif primary_type == 'performance_review':
                return 'performance_reviews'
            elif primary_type == 'resume':
                return 'resumes'
            elif primary_type == 'job_posting':
                return 'job_postings'
            elif primary_type == 'memo':
                return 'memos'
            elif primary_type == 'proposal':
                return 'proposals'
            elif primary_type == 'report':
                return 'reports'
            elif primary_type == 'receipt':
                return 'receipts'
            elif primary_type == 'financial_statement':
                return 'financial_statements'
            elif primary_type == 'nda':
                return 'ndas'
            else:
                return f"{primary_type}s"
        
        # Fallback to original logic for non-business documents
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
            # Create empty DNA for collections with no documents yet
            dna = CollectionDNA(
                core_themes=set(),
                structural_patterns=set(),
                content_markers=set(),
                quality_indicators={},
                evolution_history=[],
                compatibility_map={}
            )
            self.collection_dna[collection_name] = dna
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
    
    async def _monitor_collection_subdivision(self):
        """Monitor mature collections for subdivision opportunities."""
        
        # Get all mature collections that might be candidates for subdivision
        mature_collections = list(self.collection_registry.find({
            "status": "mature",
            "document_count": {"$gte": self.staging_threshold}  # Has enough docs to subdivide
        }))
        
        for collection_info in mature_collections:
            collection_name = collection_info['collection_name']
            
            # Skip if we recently checked this collection
            if await self._recently_checked_for_subdivision(collection_name):
                continue
            
            logger.info(f"🔍 Analyzing '{collection_name}' for subdivision opportunities...")
            
            # Analyze the collection for subdivision patterns
            subdivision_opportunities = await self._analyze_collection_for_subdivision(collection_name)
            
            if subdivision_opportunities:
                logger.info(f"📊 Found {len(subdivision_opportunities)} subdivision opportunities in '{collection_name}'")
                
                for opportunity in subdivision_opportunities:
                    await self._create_subdivision(collection_name, opportunity)
            
            # Record that we checked this collection
            await self._record_subdivision_check(collection_name)
    
    async def _recently_checked_for_subdivision(self, collection_name: str) -> bool:
        """Check if we recently analyzed this collection for subdivision."""
        from datetime import datetime, timedelta
        
        # Check if there's a recent subdivision check record
        recent_check = self.collection_registry.find_one({
            "collection_name": collection_name,
            "last_subdivision_check": {"$gte": datetime.utcnow() - timedelta(hours=1)}
        })
        
        return recent_check is not None
    
    async def _analyze_collection_for_subdivision(self, collection_name: str) -> List[Dict]:
        """Analyze a mature collection to find subdivision patterns."""
        
        collection = self.db[collection_name]
        documents = list(collection.find())
        
        if len(documents) < self.staging_threshold:
            return []
        
        subdivision_opportunities = []
        
        # Group documents by semantic sub-patterns
        subclusters = await self._find_semantic_subclusters(documents)
        
        for subcluster in subclusters:
            if len(subcluster['documents']) >= self.min_seed_size:
                
                # Generate subdivision name based on distinctive keywords
                subdivision_name = self._generate_subdivision_name(
                    parent_name=collection_name,
                    distinctive_keywords=subcluster['distinctive_keywords']
                )
                
                subdivision_opportunities.append({
                    'subdivision_name': subdivision_name,
                    'documents': subcluster['documents'],
                    'distinctive_keywords': subcluster['distinctive_keywords'],
                    'confidence': subcluster['confidence'],
                    'pattern_type': subcluster['pattern_type']
                })
        
        # Filter for high-confidence subdivisions
        high_confidence_subdivisions = [
            opp for opp in subdivision_opportunities 
            if opp['confidence'] > 0.7
        ]
        
        return high_confidence_subdivisions
    
    async def _find_semantic_subclusters(self, documents: List[Dict]) -> List[Dict]:
        """Find semantic subclusters within a collection."""
        
        subclusters = []
        
        # Group by distinctive semantic patterns
        keyword_groups = defaultdict(list)
        
        for doc in documents:
            semantic_sig = doc.get('semantic_signature', {})
            domain_keywords = semantic_sig.get('domain_keywords', [])
            
            # Find distinctive keywords (appear in some docs but not all)
            for keyword in domain_keywords:
                keyword_groups[keyword].append(doc)
        
        # Find keywords that define meaningful subgroups
        total_docs = len(documents)
        
        for keyword, docs_with_keyword in keyword_groups.items():
            keyword_frequency = len(docs_with_keyword) / total_docs
            
            # Look for keywords that split the collection meaningfully
            # (appear in 20-80% of documents - not too rare, not too common)
            if 0.2 <= keyword_frequency <= 0.8 and len(docs_with_keyword) >= self.min_seed_size:
                
                # Calculate how distinctive this keyword is
                distinctive_score = self._calculate_distinctiveness(keyword, docs_with_keyword, documents)
                
                if distinctive_score > 0.6:
                    subclusters.append({
                        'documents': docs_with_keyword,
                        'distinctive_keywords': [keyword],
                        'confidence': distinctive_score,
                        'pattern_type': 'semantic_specialization'
                    })
        
        # Also look for multi-keyword patterns (e.g., 'nfl' + 'team' vs 'mlb' + 'team')
        subclusters.extend(await self._find_multi_keyword_subclusters(documents))
        
        return subclusters
    
    async def _find_multi_keyword_subclusters(self, documents: List[Dict]) -> List[Dict]:
        """Find subclusters based on combinations of keywords."""
        
        multiclusters = []
        
        # Extract all keyword combinations from documents
        keyword_combinations = defaultdict(list)
        
        for doc in documents:
            semantic_sig = doc.get('semantic_signature', {})
            domain_keywords = set(semantic_sig.get('domain_keywords', []))
            
            # Look for specific domain patterns
            if 'nfl' in domain_keywords and 'football' in domain_keywords:
                keyword_combinations['nfl_football'].append(doc)
            elif 'mlb' in domain_keywords and 'baseball' in domain_keywords:
                keyword_combinations['mlb_baseball'].append(doc)
            elif 'medical' in domain_keywords and 'patient' in domain_keywords:
                keyword_combinations['patient_medical'].append(doc)
            elif 'expense' in domain_keywords and 'report' in domain_keywords:
                keyword_combinations['expense_reports'].append(doc)
        
        # Create subclusters for meaningful combinations
        for combo_key, combo_docs in keyword_combinations.items():
            if len(combo_docs) >= self.min_seed_size:
                
                confidence = len(combo_docs) / len(documents)
                if confidence >= 0.3:  # At least 30% of the collection
                    
                    multiclusters.append({
                        'documents': combo_docs,
                        'distinctive_keywords': combo_key.split('_'),
                        'confidence': confidence,
                        'pattern_type': 'domain_specialization'
                    })
        
        return multiclusters
    
    def _calculate_distinctiveness(self, keyword: str, docs_with_keyword: List[Dict], all_docs: List[Dict]) -> float:
        """Calculate how distinctive a keyword is for subdividing a collection."""
        
        # Base distinctiveness on frequency
        frequency_score = len(docs_with_keyword) / len(all_docs)
        
        # Penalize keywords that are too common or too rare
        if frequency_score < 0.2 or frequency_score > 0.8:
            return 0.0
        
        # Boost distinctiveness for domain-specific keywords
        domain_keywords = ['nfl', 'mlb', 'medical', 'patient', 'expense', 'technical', 'kubernetes']
        if keyword in domain_keywords:
            return min(frequency_score + 0.3, 1.0)
        
        return frequency_score
    
    def _generate_subdivision_name(self, parent_name: str, distinctive_keywords: List[str]) -> str:
        """Generate a name for a subdivision collection."""
        
        # Extract the base category from parent name
        base_parts = parent_name.split('_')
        
        if 'documents' in base_parts:
            base_parts.remove('documents')
        if 'collection' in base_parts:
            base_parts.remove('collection')
        
        # Combine distinctive keywords with parent category
        primary_keyword = distinctive_keywords[0] if distinctive_keywords else 'specialized'
        
        # Create hierarchical naming
        if len(base_parts) > 0:
            return f"{primary_keyword}_{base_parts[0]}_documents"
        else:
            return f"{primary_keyword}_documents"
    
    async def _create_subdivision(self, parent_collection: str, subdivision_info: Dict):
        """Create a new subdivision collection from a parent collection."""
        
        subdivision_name = subdivision_info['subdivision_name']
        documents_to_move = subdivision_info['documents']
        
        logger.info(f"🌿 Creating subdivision '{subdivision_name}' from '{parent_collection}' with {len(documents_to_move)} documents")
        
        # Create the new subdivision collection
        subdivision_collection = self.db[subdivision_name]
        
        # Move documents to subdivision
        for doc in documents_to_move:
            # Add subdivision metadata
            doc['parent_collection'] = parent_collection
            doc['subdivision_created_at'] = datetime.utcnow()
            doc['subdivision_reason'] = subdivision_info['pattern_type']
            
            # Remove the _id to let MongoDB generate a new one
            doc_copy = doc.copy()
            if '_id' in doc_copy:
                del doc_copy['_id']
            
            # Insert into subdivision
            subdivision_collection.insert_one(doc_copy)
            
            # Remove from parent (optional - could also mark as moved)
            parent_coll = self.db[parent_collection]
            parent_coll.delete_one({"_id": doc["_id"]})
        
        # Register the new subdivision
        await self._register_subdivision(subdivision_name, parent_collection, subdivision_info)
        
        # Update parent collection count
        await self._update_collection_count(parent_collection)
        
        logger.info(f"✅ Subdivision '{subdivision_name}' created successfully")
    
    async def _register_subdivision(self, subdivision_name: str, parent_name: str, subdivision_info: Dict):
        """Register a new subdivision in the collection registry."""
        
        subdivision_entry = {
            "collection_name": subdivision_name,
            "parent_collection": parent_name,
            "status": "mature",
            "created_at": datetime.utcnow(),
            "document_count": len(subdivision_info['documents']),
            "theme_keywords": subdivision_info['distinctive_keywords'],
            "birth_confidence": subdivision_info['confidence'],
            "subdivision_type": subdivision_info['pattern_type']
        }
        
        self.collection_registry.insert_one(subdivision_entry)
    
    async def _record_subdivision_check(self, collection_name: str):
        """Record that we checked this collection for subdivisions."""
        
        self.collection_registry.update_one(
            {"collection_name": collection_name},
            {"$set": {"last_subdivision_check": datetime.utcnow()}}
        )
    
    async def _update_collection_count(self, collection_name: str):
        """Update the document count for a collection."""
        
        collection = self.db[collection_name]
        new_count = collection.count_documents({})
        
        self.collection_registry.update_one(
            {"collection_name": collection_name},
            {"$set": {"document_count": new_count}}
        )