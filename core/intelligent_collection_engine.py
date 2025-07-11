"""
Intelligent Collection Engine

Orchestrates the entire intelligent taxonomy system.
Coordinates Claude analysis, concept drift detection, and dynamic taxonomy management
to create mindblowingly intelligent document organization.
"""

import asyncio
import json
import time
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger
import pymongo
from pathlib import Path
from bson import ObjectId

from .claude_ontology_analyzer import ClaudeOntologyAnalyzer, OntologyAnalysis
from .concept_drift_detector import ConceptDriftDetector, DriftSignal
from .dynamic_taxonomy_manager import DynamicTaxonomyManager, PlacementDecision, CollectionSpec
from .organic_collections import OrganicCollectionManager
from .config import settings

@dataclass
class ProcessingResult:
    """Result of intelligent document processing."""
    document_id: str
    collection_assigned: str
    confidence: float
    processing_time: float
    ontology_analysis: OntologyAnalysis
    drift_signal: Optional[DriftSignal]
    placement_decision: PlacementDecision
    created_collections: List[str]
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class ProcessingStats:
    """Statistics about processing performance."""
    total_documents: int
    successful_placements: int
    new_collections_created: int
    reorganizations_triggered: int
    average_confidence: float
    average_processing_time: float
    confidence_distribution: Dict[str, int]

class IntelligentCollectionEngine:
    """
    The orchestrator of intelligent document organization.
    
    Coordinates all components to create a mindblowingly intelligent
    taxonomy system that understands, adapts, and evolves.
    """
    
    def __init__(self, mongo_client: pymongo.MongoClient, db_name: str = "intelligent_data"):
        self.mongo_client = mongo_client
        self.db = mongo_client[db_name]
        self.staging_collection = self.db["document_staging"]
        
        # Initialize core components
        self.claude_analyzer = ClaudeOntologyAnalyzer()
        self.drift_detector = ConceptDriftDetector()
        self.taxonomy_manager = DynamicTaxonomyManager(mongo_client, db_name)
        self.organic_manager = OrganicCollectionManager(mongo_client, db_name)
        
        # Configuration (user's requirements)
        self.confidence_thresholds = {
            "high": 0.8,      # Auto-assign and create collections
            "medium": 0.5,    # Assign but flag for review
            "low": 0.3,       # Hold in staging
            "reorganization": 0.7  # Only reorganize when very confident
        }
        
        # Processing statistics
        self.stats = ProcessingStats(
            total_documents=0,
            successful_placements=0,
            new_collections_created=0,
            reorganizations_triggered=0,
            average_confidence=0.0,
            average_processing_time=0.0,
            confidence_distribution={"high": 0, "medium": 0, "low": 0}
        )
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the intelligent collection system."""
        logger.info("🧠 Initializing Intelligent Collection Engine")
        logger.info("🎯 Priorities: Intelligence and Accuracy over Speed")
        logger.info("📊 Volume-based triggers enabled")
        logger.info("🔄 Concept drift detection active")
        logger.info("🌟 Ready for mindblowingly intelligent document organization!")
        
    async def process_document(self, document_data: Dict[str, Any]) -> ProcessingResult:
        """
        Process a document through the complete intelligent pipeline.
        
        This is the main entry point that coordinates all components
        to achieve intelligent document organization.
        
        Args:
            document_data: Document with content, filename, etc.
            
        Returns:
            ProcessingResult with complete analysis and placement decision
        """
        start_time = time.time()
        document_id = document_data.get("_id", "unknown")
        filename = document_data.get("source_file", "unknown")
        content = document_data.get("content", "")
        
        logger.info(f"🚀 Processing document through intelligent pipeline: {filename}")
        
        try:
            # Step 1: Deep Semantic Analysis with Claude
            logger.info(f"🧠 Step 1: Claude ontological analysis...")
            ontology_analysis = await self.claude_analyzer.analyze_document(
                content, filename, str(document_id)
            )
            
            # Step 2: Concept Drift Detection  
            logger.info(f"📊 Step 2: Concept drift analysis...")
            drift_signal = await self.drift_detector.analyze_document_stream({
                "_id": document_id,
                "source_file": filename,
                "ontology_analysis": asdict(ontology_analysis),
                "timestamp": datetime.now()
            })
            
            # Step 3: Process through Organic Collection System
            logger.info(f"🌱 Step 3: Processing through organic collection system...")
            
            # Enhance document data with analysis results
            enhanced_document_data = document_data.copy()
            enhanced_document_data['semantic_signature'] = {
                'domain_keywords': [entity.entity_type for entity in ontology_analysis.entities],
                'structural_patterns': [h.medium for h in ontology_analysis.concept_hierarchies],
                'content_markers': ontology_analysis.cross_references,
                'semantic_hash': ontology_analysis.semantic_fingerprint
            }
            enhanced_document_data['ontology_analysis'] = asdict(ontology_analysis)
            enhanced_document_data['content_type'] = ontology_analysis.document_type.type_name
            enhanced_document_data['business_category'] = ontology_analysis.document_type.domain
            
            # Process through organic system (handles staging, clustering, and collection creation)
            doc_id = await self.organic_manager.process_document(enhanced_document_data)
            
            # Check if document was assigned to a collection
            staged_doc = self.staging_collection.find_one({"_id": ObjectId(doc_id)})
            
            if staged_doc and staged_doc.get('collection_assigned'):
                final_collection = staged_doc['collection_assigned']
                confidence = staged_doc.get('collection_confidence', 0.8)
                created_collections = []
            else:
                # Document is still in staging - use taxonomy manager for immediate placement
                logger.info(f"📍 Step 3b: Document in staging, evaluating immediate placement...")
                placement_decision = await self.taxonomy_manager.evaluate_document_placement(
                    ontology_analysis, drift_signal
                )
                
                # If high confidence, assign immediately
                if placement_decision.confidence >= self.confidence_thresholds["high"]:
                    final_collection = placement_decision.target_collection
                    confidence = placement_decision.confidence
                    created_collections = []
                    
                    # Update staging document with assignment
                    self.staging_collection.update_one(
                        {"_id": ObjectId(doc_id)},
                        {"$set": {
                            "collection_assigned": final_collection,
                            "collection_confidence": confidence,
                            "assigned_at": datetime.utcnow()
                        }}
                    )
                else:
                    # Keep in staging for organic clustering
                    final_collection = "document_staging"
                    confidence = placement_decision.confidence
                    created_collections = []
            
            # Step 4: Handle Concept Drift (if detected)
            if drift_signal and drift_signal.confidence >= self.confidence_thresholds["reorganization"]:
                logger.info(f"🔄 Step 4: Handling concept drift...")
                reorganization_plan = await self.drift_detector.generate_reorganization_plan(drift_signal)
                
                success = await self.taxonomy_manager.execute_reorganization(reorganization_plan)
                if success:
                    self.stats.reorganizations_triggered += 1
                    logger.info(f"✅ Successfully executed reorganization")
            
            # Step 5: Trigger organic collection evolution
            logger.info(f"🔄 Step 5: Checking for collection evolution opportunities...")
            await self.organic_manager._evolve_collections()
            await self.organic_manager._monitor_collection_subdivision()
            
            # Step 6: Update Statistics and Baselines
            processing_time = time.time() - start_time
            await self._update_processing_stats(confidence, processing_time)
            
            # Update drift detection baselines periodically
            if self.stats.total_documents % 50 == 0:
                await self.drift_detector.update_baselines()
            
            # Create processing result
            result = ProcessingResult(
                document_id=str(doc_id),
                collection_assigned=final_collection,
                confidence=confidence,
                processing_time=processing_time,
                ontology_analysis=ontology_analysis,
                drift_signal=drift_signal,
                placement_decision=placement_decision if 'placement_decision' in locals() else None,
                created_collections=created_collections,
                reasoning=self._generate_processing_reasoning(
                    ontology_analysis, drift_signal, placement_decision if 'placement_decision' in locals() else None
                ),
                metadata={
                    "filename": filename,
                    "processing_timestamp": datetime.now().isoformat(),
                    "claude_analysis_confidence": ontology_analysis.confidence_score,
                    "drift_detected": drift_signal is not None,
                    "new_collection_created": len(created_collections) > 0
                }
            )
            
            logger.info(f"✅ Successfully processed {filename} → {final_collection} "
                       f"(confidence: {confidence:.2f}, time: {processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process document {filename}: {e}")
            # Return error result
            return await self._create_error_result(document_data, str(e), time.time() - start_time)
    
    async def _assign_document_to_collection(self, document_data: Dict[str, Any],
                                           collection_name: str,
                                           ontology_analysis: OntologyAnalysis,
                                           placement_decision: PlacementDecision):
        """Assign document to target collection with enhanced metadata."""
        
        # Prepare enhanced document data
        enhanced_document = document_data.copy()
        enhanced_document.update({
            "collection_assigned": collection_name,
            "assigned_at": datetime.utcnow(),
            "collection_confidence": placement_decision.confidence,
            "ontology_analysis": asdict(ontology_analysis),
            "placement_metadata": {
                "reasoning": placement_decision.reasoning,
                "alternatives_considered": placement_decision.alternative_collections,
                "collection_created": placement_decision.create_new_collection,
                "confidence_level": self._get_confidence_level(placement_decision.confidence)
            },
            "intelligent_processing": {
                "claude_analysis_version": "1.0",
                "drift_detection_enabled": True,
                "taxonomy_management_active": True,
                "processing_pipeline": "intelligent_v1"
            }
        })
        
        # Insert into target collection
        target_collection = self.db[collection_name]
        target_collection.insert_one(enhanced_document)
        
        # Remove from staging
        if "_id" in document_data:
            self.staging_collection.delete_one({"_id": document_data["_id"]})
        
        # Update collection registry count
        self.taxonomy_manager.collection_registry.update_one(
            {"collection_name": collection_name},
            {
                "$inc": {"document_count": 1},
                "$set": {"last_modified": datetime.utcnow()}
            }
        )
        
        self.stats.successful_placements += 1
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level category."""
        if confidence >= self.confidence_thresholds["high"]:
            return "high"
        elif confidence >= self.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _generate_processing_reasoning(self, ontology_analysis: OntologyAnalysis,
                                     drift_signal: Optional[DriftSignal],
                                     placement_decision: PlacementDecision) -> str:
        """Generate human-readable reasoning for the processing decision."""
        
        reasoning_parts = []
        
        # Claude analysis reasoning
        doc_type = ontology_analysis.document_type
        reasoning_parts.append(
            f"Claude identified this as a '{doc_type.type_name}' document "
            f"in the '{doc_type.domain}' domain (confidence: {doc_type.confidence:.2f})"
        )
        
        # Entity analysis
        if ontology_analysis.entities:
            entity_types = [e.entity_type for e in ontology_analysis.entities[:3]]
            reasoning_parts.append(f"Key entities: {', '.join(entity_types)}")
        
        # Concept hierarchy
        if ontology_analysis.concept_hierarchies:
            hierarchy = ontology_analysis.concept_hierarchies[0]
            reasoning_parts.append(
                f"Abstraction path: {hierarchy.specific} → {hierarchy.medium} → {hierarchy.broad}"
            )
        
        # Drift signal impact
        if drift_signal:
            reasoning_parts.append(
                f"Concept drift detected: {drift_signal.drift_type.value} "
                f"(confidence: {drift_signal.confidence:.2f})"
            )
        
        # Placement reasoning
        reasoning_parts.append(placement_decision.reasoning)
        
        return " | ".join(reasoning_parts)
    
    async def _update_processing_stats(self, confidence: float, processing_time: float):
        """Update processing statistics."""
        self.stats.total_documents += 1
        
        # Update confidence distribution
        confidence_level = self._get_confidence_level(confidence)
        self.stats.confidence_distribution[confidence_level] += 1
        
        # Update averages
        total_docs = self.stats.total_documents
        self.stats.average_confidence = (
            (self.stats.average_confidence * (total_docs - 1) + confidence) / total_docs
        )
        self.stats.average_processing_time = (
            (self.stats.average_processing_time * (total_docs - 1) + processing_time) / total_docs
        )
    
    async def _create_error_result(self, document_data: Dict[str, Any], 
                                 error_message: str, processing_time: float) -> ProcessingResult:
        """Create error result for failed processing."""
        
        # Create minimal fallback analysis
        fallback_analysis = OntologyAnalysis(
            document_id=str(document_data.get("_id", "unknown")),
            entities=[],
            document_type=None,
            concept_hierarchies=[],
            semantic_fingerprint="error",
            cross_references=[],
            confidence_score=0.0,
            analysis_metadata={"error": error_message},
            claude_response={}
        )
        
        # Create fallback placement  
        fallback_placement = PlacementDecision(
            target_collection="error_staging",
            confidence=0.0,
            reasoning=f"Processing failed: {error_message}",
            alternative_collections=[],
            create_new_collection=False,
            new_collection_spec=None
        )
        
        return ProcessingResult(
            document_id=str(document_data.get("_id", "unknown")),
            collection_assigned="error_staging",
            confidence=0.0,
            processing_time=processing_time,
            ontology_analysis=fallback_analysis,
            drift_signal=None,
            placement_decision=fallback_placement,
            created_collections=[],
            reasoning=f"Processing failed: {error_message}",
            metadata={"error": True, "error_message": error_message}
        )
    
    async def process_document_batch(self, documents: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple documents through the intelligent pipeline."""
        logger.info(f"🚀 Processing batch of {len(documents)} documents")
        
        results = []
        for i, document in enumerate(documents):
            logger.info(f"📄 Processing document {i+1}/{len(documents)}")
            result = await self.process_document(document)
            results.append(result)
            
            # Small delay to avoid overwhelming the system  
            await asyncio.sleep(0.1)
        
        logger.info(f"✅ Completed batch processing: {len(results)} documents")
        return results
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        
        # Get taxonomy overview
        taxonomy_overview = await self.taxonomy_manager.get_taxonomy_overview()
        
        return {
            "processing_stats": asdict(self.stats),
            "taxonomy_overview": taxonomy_overview,
            "configuration": {
                "confidence_thresholds": self.confidence_thresholds,
                "volume_thresholds": self.drift_detector.volume_thresholds
            },
            "system_health": {
                "claude_analyzer_active": True,
                "drift_detector_active": True,
                "taxonomy_manager_active": True,
                "total_collections": taxonomy_overview["total_collections"]
            }
        }
    
    async def reset_database(self) -> bool:
        """
        Reset database for fresh start (user requirement).
        
        Drops all collections except configuration and starts fresh.
        """
        logger.warning("🗑️ RESETTING DATABASE FOR FRESH START")
        
        try:
            # Get all collection names
            collection_names = self.db.list_collection_names()
            
            # Drop all collections except system ones
            preserve_collections = ["collection_registry", "collection_seeds"]
            
            for collection_name in collection_names:
                if collection_name not in preserve_collections:
                    self.db.drop_collection(collection_name)
                    logger.info(f"🗑️ Dropped collection: {collection_name}")
            
            # Clear registries
            self.db["collection_registry"].delete_many({})
            self.db["collection_seeds"].delete_many({})
            
            # Reset statistics
            self.stats = ProcessingStats(
                total_documents=0,
                successful_placements=0,
                new_collections_created=0,
                reorganizations_triggered=0,
                average_confidence=0.0,
                average_processing_time=0.0,
                confidence_distribution={"high": 0, "medium": 0, "low": 0}
            )
            
            # Reset component states
            await self.drift_detector.update_baselines(force_update=True)
            
            # Recreate staging collection
            self.staging_collection = self.db["document_staging"]
            
            logger.info("✅ Database reset complete - ready for fresh intelligent organization!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database reset failed: {e}")
            return False
    
    async def validate_system_health(self) -> Dict[str, Any]:
        """Validate that all system components are working correctly."""
        health_report = {
            "overall_status": "healthy",
            "components": {},
            "issues": [],
            "recommendations": []
        }
        
        # Test Claude analyzer
        try:
            test_analysis = await self.claude_analyzer.analyze_document(
                "Test document content", "test.txt", "test_id"
            )
            health_report["components"]["claude_analyzer"] = "healthy"
        except Exception as e:
            health_report["components"]["claude_analyzer"] = f"error: {e}"
            health_report["issues"].append("Claude analyzer not responding")
        
        # Test drift detector
        try:
            test_signal = await self.drift_detector.analyze_document_stream({
                "_id": "test",
                "ontology_analysis": {},
                "timestamp": datetime.now()
            })
            health_report["components"]["drift_detector"] = "healthy"
        except Exception as e:
            health_report["components"]["drift_detector"] = f"error: {e}"
            health_report["issues"].append("Drift detector not working")
        
        # Test taxonomy manager
        try:
            taxonomy_overview = await self.taxonomy_manager.get_taxonomy_overview()
            health_report["components"]["taxonomy_manager"] = "healthy"
        except Exception as e:
            health_report["components"]["taxonomy_manager"] = f"error: {e}"
            health_report["issues"].append("Taxonomy manager not accessible")
        
        # Overall health assessment
        if health_report["issues"]:
            health_report["overall_status"] = "degraded"
        
        return health_report
    
    async def get_ecosystem_insights(self) -> Dict[str, Any]:
        """Get insights about the organic collection ecosystem."""
        return await self.organic_manager.get_collection_insights()

# Global instance (will be initialized with MongoDB client)
intelligent_collection_engine = None