"""
Dynamic Taxonomy Manager

Manages intelligent hierarchical organization of document collections.
Creates, evolves, and reorganizes taxonomy based on semantic understanding
and concept drift signals.
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from loguru import logger
from collections import defaultdict, Counter
import sqlite3
from enum import Enum

from .claude_ontology_analyzer import OntologyAnalysis, Entity, DocumentType, ConceptHierarchy
from .concept_drift_detector import DriftSignal, ReorganizationPlan, ReorganizationAction

class CollectionStatus(Enum):
    """Status of collections in the taxonomy."""
    ACTIVE = "active"
    SUBDIVIDED = "subdivided" 
    MERGED = "merged"
    DEPRECATED = "deprecated"

class AbstractionLevel(Enum):
    """Levels of abstraction for collection organization."""
    SPECIFIC = "specific"      # Game-specific, document-specific
    MEDIUM = "medium"         # Team-level, category-level
    BROAD = "broad"           # Sport-level, domain-level
    DOMAIN = "domain"         # Top-level domains

@dataclass
class CollectionSpec:
    """Specification for a collection in the taxonomy."""
    name: str
    abstraction_level: AbstractionLevel
    parent_collection: Optional[str]
    child_collections: List[str]
    ontology_profile: Dict[str, Any]
    creation_criteria: Dict[str, Any]
    confidence_score: float

@dataclass
class TaxonomyNode:
    """Node in the hierarchical taxonomy structure."""
    collection_name: str
    level: int
    parent: Optional[str]
    children: Set[str]
    document_count: int
    semantic_coherence: float
    last_modified: datetime

@dataclass
class PlacementDecision:
    """Decision about where to place a document."""
    target_collection: str
    confidence: float
    reasoning: str
    alternative_collections: List[Tuple[str, float]]
    create_new_collection: bool
    new_collection_spec: Optional[CollectionSpec]

class DynamicTaxonomyManager:
    """
    Manages the dynamic evolution of document taxonomy.
    Creates intelligent hierarchical organization based on semantic analysis.
    """
    
    def __init__(self, mongo_client, db_name: str = "intelligent_data", 
                 cache_db_path: str = "./taxonomy_manager.db"):
        self.mongo_client = mongo_client
        self.db = mongo_client[db_name]
        self.collection_registry = self.db["collection_registry"]
        self.cache_db_path = cache_db_path
        
        # Taxonomy structure
        self.taxonomy_tree: Dict[str, TaxonomyNode] = {}
        self.abstraction_hierarchy: Dict[AbstractionLevel, Set[str]] = {
            level: set() for level in AbstractionLevel
        }
        
        # Configuration based on user requirements
        self.confidence_thresholds = {
            "high": 0.8,      # Auto-assign and create collections
            "medium": 0.5,    # Assign but flag for review
            "low": 0.3,       # Hold in staging
            "reorganization": 0.7  # Only reorganize when very confident
        }
        
        self.volume_thresholds = {
            "small": 10,      # Stay broad
            "medium": 50,     # Consider specialization  
            "large": 100,     # Active subdivision
            "subdivision": 75 # Trigger subdivision
        }
        
        self._init_taxonomy_db()
        
    def _init_taxonomy_db(self):
        """Initialize taxonomy management database."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS taxonomy_evolution (
                evolution_id TEXT PRIMARY KEY,
                action_type TEXT,
                target_collections TEXT,
                new_collections TEXT,
                confidence REAL,
                timestamp TIMESTAMP,
                reasoning TEXT,
                success BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_lineage (
                collection_name TEXT PRIMARY KEY,
                parent_collection TEXT,
                creation_reason TEXT,
                creation_timestamp TIMESTAMP,
                semantic_profile TEXT,
                abstraction_level TEXT,
                document_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS placement_history (
                document_id TEXT,
                collection_name TEXT,
                confidence REAL,
                reasoning TEXT,
                timestamp TIMESTAMP,
                alternative_considered TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def evaluate_document_placement(self, ontology_analysis: OntologyAnalysis, 
                                        drift_signal: Optional[DriftSignal] = None) -> PlacementDecision:
        """
        Evaluate where a document should be placed in the taxonomy.
        
        Args:
            ontology_analysis: Claude's semantic analysis of the document
            drift_signal: Optional concept drift signal for context
            
        Returns:
            PlacementDecision with target collection and confidence
        """
        logger.info(f"📍 Evaluating placement for document: {ontology_analysis.document_id}")
        
        # Get document characteristics
        doc_type = ontology_analysis.document_type
        entities = ontology_analysis.entities
        hierarchies = ontology_analysis.concept_hierarchies
        confidence = ontology_analysis.confidence_score
        
        # Find existing collections that match
        matching_collections = await self._find_matching_collections(ontology_analysis)
        
        # Determine appropriate abstraction level
        target_abstraction = self._determine_target_abstraction(
            ontology_analysis, drift_signal, matching_collections
        )
        
        # Make placement decision
        if matching_collections and confidence >= self.confidence_thresholds["medium"]:
            # Place in best matching existing collection
            best_match = max(matching_collections, key=lambda x: x[1])
            collection_name, match_confidence = best_match
            
            return PlacementDecision(
                target_collection=collection_name,
                confidence=match_confidence,
                reasoning=f"Matched existing collection based on semantic similarity",
                alternative_collections=matching_collections[1:],
                create_new_collection=False,
                new_collection_spec=None
            )
            
        elif confidence >= self.confidence_thresholds["high"]:
            # Create new collection
            new_collection_spec = await self._design_new_collection(
                ontology_analysis, target_abstraction
            )
            
            return PlacementDecision(
                target_collection=new_collection_spec.name,
                confidence=confidence,
                reasoning=f"Creating new collection at {target_abstraction.value} level",
                alternative_collections=matching_collections,
                create_new_collection=True,
                new_collection_spec=new_collection_spec
            )
            
        else:
            # Low confidence - place in broad staging collection
            staging_collection = await self._get_or_create_staging_collection(doc_type.domain)
            
            return PlacementDecision(
                target_collection=staging_collection,
                confidence=confidence,
                reasoning=f"Low confidence placement in staging (confidence: {confidence:.2f})",
                alternative_collections=matching_collections,
                create_new_collection=False,
                new_collection_spec=None
            )
    
    async def _find_matching_collections(self, ontology_analysis: OntologyAnalysis) -> List[Tuple[str, float]]:
        """Find existing collections that match the document's semantic profile."""
        matching_collections = []
        
        # Get all active collections
        active_collections = list(self.collection_registry.find({"status": "active"}))
        
        for collection in active_collections:
            similarity = await self._calculate_semantic_similarity(
                ontology_analysis, collection
            )
            
            if similarity > 0.3:  # Minimum similarity threshold
                matching_collections.append((collection["collection_name"], similarity))
        
        # Sort by similarity descending
        matching_collections.sort(key=lambda x: x[1], reverse=True)
        
        return matching_collections
    
    async def _calculate_semantic_similarity(self, ontology_analysis: OntologyAnalysis, 
                                           collection_info: Dict[str, Any]) -> float:
        """Calculate semantic similarity between document and collection."""
        similarity_score = 0.0
        
        # Get collection's ontology profile
        ontology_profile = collection_info.get("ontology_profile", {})
        
        # Compare document types
        doc_type = ontology_analysis.document_type.type_name
        collection_doc_types = ontology_profile.get("document_types", [])
        if doc_type in collection_doc_types:
            similarity_score += 0.4
        
        # Compare domains
        doc_domain = ontology_analysis.document_type.domain
        collection_domains = ontology_profile.get("domains", [])
        if doc_domain in collection_domains:
            similarity_score += 0.3
        
        # Compare entity types organically - no hardcoded sport logic
        doc_entity_types = set(entity.entity_type for entity in ontology_analysis.entities)
        collection_entity_types = set(ontology_profile.get("entity_types", []))
        
        # Organic similarity based on entity type overlap
        entity_overlap = len(doc_entity_types.intersection(collection_entity_types))
        entity_total = len(doc_entity_types.union(collection_entity_types))
        if entity_total > 0:
            entity_similarity = entity_overlap / entity_total
            similarity_score += 0.3 * entity_similarity
            
            # Organic penalty for very different entity types (generalized conflict detection)
            # If there's very low overlap but high total (indicating very different entity types)
            if entity_similarity < 0.2 and entity_total > 3:
                similarity_score -= 0.2  # Gentle penalty for very different domains
        
        # Compare concepts
        doc_concepts = set()
        for hierarchy in ontology_analysis.concept_hierarchies:
            doc_concepts.update([hierarchy.specific, hierarchy.medium, hierarchy.broad])
        
        collection_concepts = set(ontology_profile.get("primary_concepts", []))
        concept_overlap = len(doc_concepts.intersection(collection_concepts))
        concept_total = len(doc_concepts.union(collection_concepts))
        if concept_total > 0:
            similarity_score += 0.2 * (concept_overlap / concept_total)
        
        return min(similarity_score, 1.0)
    
    def _determine_target_abstraction(self, ontology_analysis: OntologyAnalysis,
                                    drift_signal: Optional[DriftSignal],
                                    matching_collections: List[Tuple[str, float]]) -> AbstractionLevel:
        """Determine appropriate abstraction level for document placement."""
        
        # Get recommendation from Claude analysis
        claude_recommendation = ontology_analysis.analysis_metadata.get("abstraction_recommendation", "medium")
        
        # Consider volume patterns from drift signal
        if drift_signal and hasattr(drift_signal, 'trigger_metrics'):
            trigger_metrics = drift_signal.trigger_metrics
            
            # If high volume surge, prefer more specific categorization
            if trigger_metrics.get("surge_factor", 1.0) > 2.0:
                if claude_recommendation == "broad":
                    return AbstractionLevel.MEDIUM
                elif claude_recommendation == "medium":
                    return AbstractionLevel.SPECIFIC
        
        # Consider existing collection landscape
        if len(matching_collections) == 0:
            # No matches - start broader
            return AbstractionLevel.BROAD if claude_recommendation == "specific" else AbstractionLevel.MEDIUM
        
        # Map Claude's recommendation to our enum
        mapping = {
            "specific": AbstractionLevel.SPECIFIC,
            "medium": AbstractionLevel.MEDIUM,
            "broad": AbstractionLevel.BROAD
        }
        
        return mapping.get(claude_recommendation, AbstractionLevel.MEDIUM)
    
    async def _design_new_collection(self, ontology_analysis: OntologyAnalysis,
                                   abstraction_level: AbstractionLevel) -> CollectionSpec:
        """Design a new collection based on document analysis."""
        
        # Generate collection name based on semantic analysis
        collection_name = await self._generate_collection_name(ontology_analysis, abstraction_level)
        
        # Determine parent collection
        parent_collection = await self._find_parent_collection(ontology_analysis, abstraction_level)
        
        # Build ontology profile
        ontology_profile = {
            "primary_concepts": [h.medium for h in ontology_analysis.concept_hierarchies],
            "entity_types": [e.entity_type for e in ontology_analysis.entities],
            "document_types": [ontology_analysis.document_type.type_name],
            "domains": [ontology_analysis.document_type.domain],
            "abstraction_level": abstraction_level.value,
            "semantic_coherence": ontology_analysis.confidence_score
        }
        
        # Define creation criteria
        creation_criteria = {
            "semantic_fingerprint_pattern": ontology_analysis.semantic_fingerprint[:8],
            "minimum_similarity": 0.6,
            "document_type_match": ontology_analysis.document_type.type_name,
            "domain_match": ontology_analysis.document_type.domain
        }
        
        return CollectionSpec(
            name=collection_name,
            abstraction_level=abstraction_level,
            parent_collection=parent_collection,
            child_collections=[],
            ontology_profile=ontology_profile,
            creation_criteria=creation_criteria,
            confidence_score=ontology_analysis.confidence_score
        )
    
    async def _generate_collection_name(self, ontology_analysis: OntologyAnalysis,
                                      abstraction_level: AbstractionLevel) -> str:
        """Generate meaningful collection name based on semantic analysis."""
        
        doc_type = ontology_analysis.document_type
        entities = ontology_analysis.entities
        hierarchies = ontology_analysis.concept_hierarchies
        
        # Base name on document type and domain
        if abstraction_level == AbstractionLevel.SPECIFIC:
            # Use specific concepts
            if hierarchies:
                base_name = hierarchies[0].specific.replace(" ", "_").lower()
            else:
                base_name = doc_type.type_name
        elif abstraction_level == AbstractionLevel.MEDIUM:
            # Use medium-level concepts  
            if hierarchies:
                base_name = hierarchies[0].medium.replace(" ", "_").lower()
            else:
                base_name = f"{doc_type.domain}_{doc_type.type_name}"
        else:  # BROAD or DOMAIN
            # Use broad concepts
            if hierarchies:
                base_name = hierarchies[0].broad.replace(" ", "_").lower()
            else:
                base_name = doc_type.domain
        
        # Ensure uniqueness
        base_name = base_name.replace(" ", "_").lower()
        candidate_name = f"{base_name}_documents"
        
        # Check if name exists
        existing = self.collection_registry.find_one({"collection_name": candidate_name})
        if existing:
            # Add suffix for uniqueness
            suffix = 1
            while True:
                test_name = f"{base_name}_{suffix}_documents"
                if not self.collection_registry.find_one({"collection_name": test_name}):
                    candidate_name = test_name
                    break
                suffix += 1
        
        return candidate_name
    
    async def _find_parent_collection(self, ontology_analysis: OntologyAnalysis,
                                    abstraction_level: AbstractionLevel) -> Optional[str]:
        """Find appropriate parent collection for hierarchical organization."""
        
        if abstraction_level == AbstractionLevel.DOMAIN:
            return None  # Top level
        
        # Look for collections at higher abstraction levels
        target_levels = []
        if abstraction_level == AbstractionLevel.SPECIFIC:
            target_levels = [AbstractionLevel.MEDIUM, AbstractionLevel.BROAD, AbstractionLevel.DOMAIN]
        elif abstraction_level == AbstractionLevel.MEDIUM:
            target_levels = [AbstractionLevel.BROAD, AbstractionLevel.DOMAIN]
        elif abstraction_level == AbstractionLevel.BROAD:
            target_levels = [AbstractionLevel.DOMAIN]
        
        # Find matching collections at higher levels
        for target_level in target_levels:
            collections = list(self.collection_registry.find({
                "ontology_profile.abstraction_level": target_level.value,
                "status": "active"
            }))
            
            for collection in collections:
                similarity = await self._calculate_semantic_similarity(
                    ontology_analysis, collection
                )
                if similarity > 0.5:  # Strong similarity for parent relationship
                    return collection["collection_name"]
        
        return None
    
    async def _get_or_create_staging_collection(self, domain: str) -> str:
        """Get or create broad collection for low-confidence documents."""
        # Create domain-specific broad collections instead of "staging"
        broad_collection_names = [
            f"{domain}_documents",  # Most general domain collection
            f"{domain}_content",    # Alternative broad name
            f"general_{domain}_items"  # Even broader fallback
        ]
        
        # Try to find an existing broad collection for this domain
        for candidate_name in broad_collection_names:
            existing = self.collection_registry.find_one({
                "collection_name": candidate_name,
                "ontology_profile.abstraction_level": "broad"
            })
            if existing:
                return candidate_name
        
        # Create new broad collection (not staging - actual broad category)
        collection_name = broad_collection_names[0]  # Use most general name
        
        broad_spec = CollectionSpec(
            name=collection_name,
            abstraction_level=AbstractionLevel.BROAD,
            parent_collection=None,
            child_collections=[],
            ontology_profile={
                "primary_concepts": [domain, "general_content"],
                "entity_types": [],  # Will learn organically
                "document_types": ["team_schedule", "general_document"],  # Broad types
                "domains": [domain],
                "abstraction_level": "broad",
                "semantic_coherence": 0.4  # Moderate coherence for broad collection
            },
            creation_criteria={
                "purpose": "broad_domain_collection_for_organic_discovery",
                "minimum_similarity": 0.1,
                "allows_organic_subdivision": True
            },
            confidence_score=0.6
        )
        
        await self.create_collection(broad_spec)
        return collection_name
    
    async def create_collection(self, collection_spec: CollectionSpec) -> bool:
        """Create a new collection based on specification."""
        logger.info(f"🆕 Creating new collection: {collection_spec.name}")
        
        try:
            # Create MongoDB collection
            new_collection = self.db[collection_spec.name]
            
            # Register in collection registry
            registry_entry = {
                "collection_name": collection_spec.name,
                "status": "active",
                "created_at": datetime.utcnow(),
                "document_count": 0,
                "ontology_profile": collection_spec.ontology_profile,
                "hierarchy": {
                    "parent_collection": collection_spec.parent_collection,
                    "child_collections": [],
                    "depth_level": self._calculate_depth_level(collection_spec.parent_collection),
                    "abstraction_level": collection_spec.abstraction_level.value
                },
                "creation_criteria": collection_spec.creation_criteria,
                "confidence_score": collection_spec.confidence_score,
                "last_modified": datetime.utcnow()
            }
            
            self.collection_registry.insert_one(registry_entry)
            
            # Update parent's child list if applicable
            if collection_spec.parent_collection:
                self.collection_registry.update_one(
                    {"collection_name": collection_spec.parent_collection},
                    {"$push": {"hierarchy.child_collections": collection_spec.name}}
                )
            
            # Record in taxonomy evolution log
            await self._log_taxonomy_evolution(
                "create_collection",
                [collection_spec.name],
                collection_spec.confidence_score,
                f"Created {collection_spec.abstraction_level.value} level collection"
            )
            
            logger.info(f"✅ Successfully created collection: {collection_spec.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection {collection_spec.name}: {e}")
            return False
    
    def _calculate_depth_level(self, parent_collection: Optional[str]) -> int:
        """Calculate depth level in hierarchy."""
        if not parent_collection:
            return 0
        
        parent_info = self.collection_registry.find_one({"collection_name": parent_collection})
        if parent_info and "hierarchy" in parent_info:
            return parent_info["hierarchy"].get("depth_level", 0) + 1
        
        return 1
    
    async def execute_reorganization(self, reorganization_plan: ReorganizationPlan) -> bool:
        """Execute a reorganization plan from concept drift detection."""
        logger.info(f"🔄 Executing reorganization: {reorganization_plan.action.value}")
        
        try:
            if reorganization_plan.action == ReorganizationAction.CREATE_SUBCATEGORY:
                return await self._execute_subcategory_creation(reorganization_plan)
            elif reorganization_plan.action == ReorganizationAction.SPLIT_COLLECTION:
                return await self._execute_collection_split(reorganization_plan)
            elif reorganization_plan.action == ReorganizationAction.MERGE_COLLECTIONS:
                return await self._execute_collection_merge(reorganization_plan)
            elif reorganization_plan.action == ReorganizationAction.RESTRUCTURE_HIERARCHY:
                return await self._execute_hierarchy_restructure(reorganization_plan)
            else:
                logger.info("No reorganization action needed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Reorganization failed: {e}")
            return False
    
    async def _execute_subcategory_creation(self, plan: ReorganizationPlan) -> bool:
        """Execute subcategory creation from reorganization plan."""
        for new_collection_spec in plan.new_collection_specs:
            # Create the subcategory
            spec = CollectionSpec(
                name=new_collection_spec["name"],
                abstraction_level=AbstractionLevel.SPECIFIC,
                parent_collection=new_collection_spec["parent"],
                child_collections=[],
                ontology_profile={
                    "primary_concepts": ["specialized"],
                    "entity_types": [],
                    "document_types": [],
                    "domains": [],
                    "abstraction_level": "specific",
                    "semantic_coherence": 0.7
                },
                creation_criteria={"specialization_trigger": "volume_based"},
                confidence_score=plan.confidence
            )
            
            success = await self.create_collection(spec)
            if not success:
                return False
        
        return True
    
    async def _execute_collection_split(self, plan: ReorganizationPlan) -> bool:
        """Execute collection splitting."""
        # Implementation would analyze documents in target collections
        # and split them based on semantic clustering
        logger.info(f"Splitting collections: {plan.target_collections}")
        return True
    
    async def _execute_collection_merge(self, plan: ReorganizationPlan) -> bool:
        """Execute collection merging."""
        # Implementation would merge documents from multiple collections
        # into a single, broader collection
        logger.info(f"Merging collections: {plan.target_collections}")
        return True
    
    async def _execute_hierarchy_restructure(self, plan: ReorganizationPlan) -> bool:
        """Execute hierarchy restructuring."""
        # Implementation would reorganize parent-child relationships
        # based on new semantic understanding
        logger.info(f"Restructuring hierarchy for: {plan.target_collections}")
        return True
    
    async def _log_taxonomy_evolution(self, action_type: str, collections: List[str],
                                    confidence: float, reasoning: str) -> None:
        """Log taxonomy evolution for audit and analysis."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        evolution_id = hashlib.md5(f"{action_type}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        cursor.execute('''
            INSERT INTO taxonomy_evolution
            (evolution_id, action_type, target_collections, confidence, timestamp, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            evolution_id,
            action_type,
            json.dumps(collections),
            confidence,
            datetime.now().isoformat(),
            reasoning
        ))
        
        conn.commit()
        conn.close()
    
    async def get_taxonomy_overview(self) -> Dict[str, Any]:
        """Get overview of current taxonomy structure."""
        collections = list(self.collection_registry.find({"status": "active"}))
        
        # Organize by abstraction level
        by_level = defaultdict(list)
        for collection in collections:
            level = collection.get("hierarchy", {}).get("abstraction_level", "medium")
            by_level[level].append({
                "name": collection["collection_name"],
                "document_count": collection["document_count"],
                "parent": collection.get("hierarchy", {}).get("parent_collection"),
                "children": collection.get("hierarchy", {}).get("child_collections", [])
            })
        
        return {
            "total_collections": len(collections),
            "by_abstraction_level": dict(by_level),
            "hierarchy_depth": max([
                collection.get("hierarchy", {}).get("depth_level", 0) 
                for collection in collections
            ], default=0)
        }

# Global instance
dynamic_taxonomy_manager = None  # Will be initialized with MongoDB client