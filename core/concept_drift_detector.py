"""
Concept Drift Detection System

Monitors document streams for pattern changes and triggers intelligent
reorganization based on volume, semantic shifts, and diversity metrics.
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple, Deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from loguru import logger
from collections import deque, defaultdict, Counter
import sqlite3
from enum import Enum

class DriftType(Enum):
    """Types of concept drift that can be detected."""
    VOLUME_SURGE = "volume_surge"
    SEMANTIC_SHIFT = "semantic_shift"
    DIVERSITY_CHANGE = "diversity_change"
    PATTERN_EMERGENCE = "pattern_emergence"
    DOMAIN_EXPANSION = "domain_expansion"

class ReorganizationAction(Enum):
    """Types of reorganization actions that can be recommended."""
    CREATE_SUBCATEGORY = "create_subcategory"
    MERGE_COLLECTIONS = "merge_collections"
    SPLIT_COLLECTION = "split_collection"
    RESTRUCTURE_HIERARCHY = "restructure_hierarchy"
    NO_ACTION = "no_action"

@dataclass
class DocumentStreamMetrics:
    """Metrics about the current document stream."""
    total_documents: int
    documents_in_window: int
    unique_document_types: int
    unique_domains: int
    semantic_diversity: float
    average_confidence: float
    dominant_patterns: List[str]
    emerging_patterns: List[str]

@dataclass
class DriftSignal:
    """Signal indicating concept drift detection."""
    drift_type: DriftType
    confidence: float
    severity: float  # 0.0 = minor, 1.0 = major
    affected_collections: List[str]
    recommended_action: ReorganizationAction
    trigger_metrics: Dict[str, Any]
    timestamp: datetime
    description: str

@dataclass
class VolumePattern:
    """Pattern analysis for document volume."""
    current_rate: float  # docs per time unit
    baseline_rate: float
    surge_factor: float  # current/baseline ratio
    trend: str  # "increasing", "decreasing", "stable"
    collections_affected: List[str]

@dataclass
class SemanticPattern:
    """Pattern analysis for semantic content."""
    dominant_concepts: List[str]
    emerging_concepts: List[str]
    concept_shift_magnitude: float
    new_domains_detected: List[str]
    entity_type_distribution: Dict[str, int]

@dataclass
class ReorganizationPlan:
    """Plan for reorganizing collections based on drift detection."""
    action: ReorganizationAction
    target_collections: List[str]
    new_collection_specs: List[Dict[str, Any]]
    migration_strategy: str
    confidence: float
    estimated_impact: str
    rollback_plan: Dict[str, Any]

class ConceptDriftDetector:
    """
    Detects concept drift in document streams and recommends reorganization actions.
    Uses volume-based triggers, semantic analysis, and diversity metrics.
    """
    
    def __init__(self, window_size: int = 100, db_path: str = "./concept_drift.db"):
        self.window_size = window_size
        self.db_path = db_path
        
        # Rolling window for document analysis
        self.document_window: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.semantic_window: Deque[str] = deque(maxlen=window_size)
        
        # Baseline patterns for comparison
        self.baseline_patterns = {}
        self.collection_baselines = {}
        
        # Configuration for volume-based triggers (from user requirements)
        self.volume_thresholds = {
            "small_collection": 10,      # Stay broad
            "medium_collection": 50,     # Consider specialization
            "large_collection": 100,     # Active subdivision
            "subdivision_trigger": 75    # Volume to trigger subdivision
        }
        
        # Drift detection parameters
        self.drift_sensitivity = 0.3  # Lower = more sensitive
        self.confidence_threshold = 0.7  # Minimum confidence for reorganization
        
        self._init_drift_db()
        
    def _init_drift_db(self):
        """Initialize database for drift detection tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_events (
                event_id TEXT PRIMARY KEY,
                drift_type TEXT,
                confidence REAL,
                severity REAL,
                affected_collections TEXT,
                recommended_action TEXT,
                trigger_metrics TEXT,
                timestamp TIMESTAMP,
                description TEXT,
                executed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS volume_baselines (
                collection_name TEXT PRIMARY KEY,
                baseline_rate REAL,
                last_updated TIMESTAMP,
                document_count INTEGER,
                pattern_stability REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_baselines (
                pattern_hash TEXT PRIMARY KEY,
                pattern_description TEXT,
                frequency INTEGER,
                collections TEXT,
                stability_score REAL,
                last_seen TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def analyze_document_stream(self, new_document: Dict[str, Any]) -> Optional[DriftSignal]:
        """
        Analyze incoming document for concept drift patterns.
        
        Args:
            new_document: Document with ontology analysis results
            
        Returns:
            DriftSignal if significant drift detected, None otherwise
        """
        # Add document to rolling windows
        self.document_window.append(new_document)
        
        if "ontology_analysis" in new_document:
            semantic_fingerprint = new_document["ontology_analysis"].get("semantic_fingerprint", "")
            self.semantic_window.append(semantic_fingerprint)
        
        # Only analyze if we have sufficient data
        if len(self.document_window) < 10:
            return None
            
        logger.debug(f"🔍 Analyzing concept drift for document stream (window size: {len(self.document_window)})")
        
        # Analyze different types of drift
        volume_drift = await self._detect_volume_drift()
        semantic_drift = await self._detect_semantic_drift()
        diversity_drift = await self._detect_diversity_drift()
        
        # Determine most significant drift
        drift_signals = [volume_drift, semantic_drift, diversity_drift]
        significant_drifts = [d for d in drift_signals if d and d.confidence > self.confidence_threshold]
        
        if significant_drifts:
            # Return highest confidence drift
            return max(significant_drifts, key=lambda x: x.confidence)
        
        return None
    
    async def _detect_volume_drift(self) -> Optional[DriftSignal]:
        """Detect volume-based concept drift (user's primary requirement)."""
        if len(self.document_window) < 20:
            return None
            
        # Analyze recent volume patterns
        volume_pattern = self._analyze_volume_pattern()
        
        # Check for volume surge trigger
        if volume_pattern.surge_factor > 2.0:  # 2x normal rate
            return DriftSignal(
                drift_type=DriftType.VOLUME_SURGE,
                confidence=min(volume_pattern.surge_factor / 3.0, 1.0),
                severity=min(volume_pattern.surge_factor / 5.0, 1.0),
                affected_collections=volume_pattern.collections_affected,
                recommended_action=self._recommend_volume_action(volume_pattern),
                trigger_metrics={
                    "surge_factor": volume_pattern.surge_factor,
                    "current_rate": volume_pattern.current_rate,
                    "baseline_rate": volume_pattern.baseline_rate,
                    "trend": volume_pattern.trend
                },
                timestamp=datetime.now(),
                description=f"Volume surge detected: {volume_pattern.surge_factor:.1f}x normal rate"
            )
        
        # Check for collection size-based triggers (user's volume requirements)
        collection_sizes = self._get_current_collection_sizes()
        
        for collection_name, size in collection_sizes.items():
            if size >= self.volume_thresholds["subdivision_trigger"]:
                return DriftSignal(
                    drift_type=DriftType.VOLUME_SURGE,
                    confidence=0.8,
                    severity=0.6,
                    affected_collections=[collection_name],
                    recommended_action=ReorganizationAction.CREATE_SUBCATEGORY,
                    trigger_metrics={
                        "collection_size": size,
                        "subdivision_threshold": self.volume_thresholds["subdivision_trigger"]
                    },
                    timestamp=datetime.now(),
                    description=f"Collection '{collection_name}' reached subdivision threshold ({size} docs)"
                )
        
        return None
    
    async def _detect_semantic_drift(self) -> Optional[DriftSignal]:
        """Detect semantic concept drift in document types."""
        if len(self.semantic_window) < 30:
            return None
            
        # Analyze semantic patterns
        semantic_pattern = self._analyze_semantic_patterns()
        
        # Check for significant semantic shift
        if semantic_pattern.concept_shift_magnitude > 0.5:
            return DriftSignal(
                drift_type=DriftType.SEMANTIC_SHIFT,
                confidence=semantic_pattern.concept_shift_magnitude,
                severity=semantic_pattern.concept_shift_magnitude * 0.8,
                affected_collections=list(semantic_pattern.entity_type_distribution.keys()),
                recommended_action=ReorganizationAction.RESTRUCTURE_HIERARCHY,
                trigger_metrics={
                    "shift_magnitude": semantic_pattern.concept_shift_magnitude,
                    "new_domains": semantic_pattern.new_domains_detected,
                    "emerging_concepts": semantic_pattern.emerging_concepts
                },
                timestamp=datetime.now(),
                description=f"Semantic shift detected: new concepts emerging"
            )
        
        # Check for new domain emergence
        if semantic_pattern.new_domains_detected:
            return DriftSignal(
                drift_type=DriftType.DOMAIN_EXPANSION,
                confidence=0.7,
                severity=0.5,
                affected_collections=[],
                recommended_action=ReorganizationAction.CREATE_SUBCATEGORY,
                trigger_metrics={
                    "new_domains": semantic_pattern.new_domains_detected,
                    "emerging_concepts": semantic_pattern.emerging_concepts
                },
                timestamp=datetime.now(),
                description=f"New domains detected: {', '.join(semantic_pattern.new_domains_detected)}"
            )
        
        return None
    
    async def _detect_diversity_drift(self) -> Optional[DriftSignal]:
        """Detect changes in document diversity patterns."""
        if len(self.document_window) < 25:
            return None
            
        # Calculate current diversity metrics
        current_metrics = self._calculate_stream_metrics()
        
        # Compare with baseline if available
        baseline_diversity = self.baseline_patterns.get("semantic_diversity", 0.5)
        diversity_change = abs(current_metrics.semantic_diversity - baseline_diversity)
        
        if diversity_change > 0.3:  # Significant diversity change
            action = (ReorganizationAction.SPLIT_COLLECTION if 
                     current_metrics.semantic_diversity > baseline_diversity else
                     ReorganizationAction.MERGE_COLLECTIONS)
            
            return DriftSignal(
                drift_type=DriftType.DIVERSITY_CHANGE,
                confidence=min(diversity_change * 2.0, 1.0),
                severity=diversity_change,
                affected_collections=[],
                recommended_action=action,
                trigger_metrics={
                    "current_diversity": current_metrics.semantic_diversity,
                    "baseline_diversity": baseline_diversity,
                    "diversity_change": diversity_change
                },
                timestamp=datetime.now(),
                description=f"Document diversity changed significantly: {diversity_change:.2f}"
            )
        
        return None
    
    def _analyze_volume_pattern(self) -> VolumePattern:
        """Analyze volume patterns in the document window."""
        if len(self.document_window) < 10:
            return VolumePattern(0, 0, 1.0, "stable", [])
        
        # Calculate current rate (docs per hour, estimated)
        recent_docs = list(self.document_window)[-20:]  # Last 20 docs
        time_span = 1.0  # Assume 1 hour for estimation
        current_rate = len(recent_docs) / time_span
        
        # Get baseline rate
        baseline_rate = self.baseline_patterns.get("volume_rate", current_rate)
        if baseline_rate == 0:
            baseline_rate = 1.0
            
        surge_factor = current_rate / baseline_rate
        
        # Determine trend
        if len(self.document_window) >= 40:
            first_half = len(list(self.document_window)[:20])
            second_half = len(list(self.document_window)[-20:])
            if second_half > first_half * 1.2:
                trend = "increasing"
            elif second_half < first_half * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return VolumePattern(
            current_rate=current_rate,
            baseline_rate=baseline_rate,
            surge_factor=surge_factor,
            trend=trend,
            collections_affected=[]  # Would need collection info to populate
        )
    
    def _analyze_semantic_patterns(self) -> SemanticPattern:
        """Analyze semantic patterns in the document window."""
        if not self.document_window:
            return SemanticPattern([], [], 0.0, [], {})
        
        # Extract concepts from recent documents
        all_concepts = []
        entity_types = []
        domains = []
        
        for doc in self.document_window:
            if "ontology_analysis" in doc:
                analysis = doc["ontology_analysis"]
                
                # Extract concepts from hierarchies
                if "concept_hierarchies" in analysis:
                    for hierarchy in analysis["concept_hierarchies"]:
                        if isinstance(hierarchy, dict):
                            all_concepts.extend([
                                hierarchy.get("specific", ""),
                                hierarchy.get("medium", ""),
                                hierarchy.get("broad", "")
                            ])
                            domains.append(hierarchy.get("domain", ""))
                
                # Extract entity types
                if "entities" in analysis:
                    for entity in analysis["entities"]:
                        if isinstance(entity, dict):
                            entity_types.append(entity.get("type", ""))
        
        # Analyze concept patterns
        concept_counter = Counter([c for c in all_concepts if c])
        entity_counter = Counter([e for e in entity_types if e])
        domain_counter = Counter([d for d in domains if d])
        
        # Identify dominant and emerging patterns
        dominant_concepts = [concept for concept, count in concept_counter.most_common(5)]
        
        # Calculate shift magnitude (simplified)
        baseline_concepts = set(self.baseline_patterns.get("concepts", []))
        current_concepts = set(dominant_concepts)
        
        if baseline_concepts:
            overlap = len(baseline_concepts.intersection(current_concepts))
            total = len(baseline_concepts.union(current_concepts))
            shift_magnitude = 1.0 - (overlap / max(total, 1))
        else:
            shift_magnitude = 0.0
        
        # Detect new domains
        baseline_domains = set(self.baseline_patterns.get("domains", []))
        current_domains = set(domain_counter.keys())
        new_domains = list(current_domains - baseline_domains)
        
        return SemanticPattern(
            dominant_concepts=dominant_concepts,
            emerging_concepts=[],  # Would need temporal analysis
            concept_shift_magnitude=shift_magnitude,
            new_domains_detected=new_domains,
            entity_type_distribution=dict(entity_counter)
        )
    
    def _calculate_stream_metrics(self) -> DocumentStreamMetrics:
        """Calculate comprehensive metrics for the current document stream."""
        if not self.document_window:
            return DocumentStreamMetrics(0, 0, 0, 0, 0.0, 0.0, [], [])
        
        documents = list(self.document_window)
        
        # Basic counts
        total_docs = len(documents)
        
        # Extract document types and domains
        doc_types = set()
        domains = set()
        confidences = []
        
        for doc in documents:
            if "ontology_analysis" in doc:
                analysis = doc["ontology_analysis"]
                
                if "document_type" in analysis:
                    doc_type_info = analysis["document_type"]
                    if isinstance(doc_type_info, dict):
                        doc_types.add(doc_type_info.get("type", ""))
                        domains.add(doc_type_info.get("domain", ""))
                        confidences.append(doc_type_info.get("confidence", 0.5))
        
        # Calculate diversity (simplified Shannon entropy)
        type_counts = Counter()
        for doc in documents:
            if "ontology_analysis" in doc:
                analysis = doc["ontology_analysis"]
                if "document_type" in analysis and isinstance(analysis["document_type"], dict):
                    doc_type = analysis["document_type"].get("type", "unknown")
                    type_counts[doc_type] += 1
        
        diversity = self._calculate_shannon_entropy(list(type_counts.values()))
        
        return DocumentStreamMetrics(
            total_documents=total_docs,
            documents_in_window=total_docs,
            unique_document_types=len(doc_types),
            unique_domains=len(domains),
            semantic_diversity=diversity,
            average_confidence=np.mean(confidences) if confidences else 0.5,
            dominant_patterns=list(doc_types)[:3],
            emerging_patterns=[]  # Would need temporal analysis
        )
    
    def _calculate_shannon_entropy(self, counts: List[int]) -> float:
        """Calculate Shannon entropy for diversity measurement."""
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        entropy = 0.0
        
        for count in counts:
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _recommend_volume_action(self, volume_pattern: VolumePattern) -> ReorganizationAction:
        """Recommend action based on volume pattern analysis."""
        if volume_pattern.surge_factor > 3.0:
            return ReorganizationAction.CREATE_SUBCATEGORY
        elif volume_pattern.surge_factor > 2.0:
            return ReorganizationAction.SPLIT_COLLECTION
        else:
            return ReorganizationAction.NO_ACTION
    
    def _get_current_collection_sizes(self) -> Dict[str, int]:
        """Get current collection sizes for volume analysis."""
        # This would integrate with the collection registry
        # For now, return estimated sizes based on document window
        collection_sizes = defaultdict(int)
        
        for doc in self.document_window:
            collection = doc.get("collection_assigned", "unknown")
            collection_sizes[collection] += 1
        
        return dict(collection_sizes)
    
    async def update_baselines(self, force_update: bool = False):
        """Update baseline patterns for drift comparison."""
        if len(self.document_window) < 50 and not force_update:
            return
        
        logger.info("📊 Updating concept drift baselines")
        
        # Update volume baseline
        volume_pattern = self._analyze_volume_pattern()
        self.baseline_patterns["volume_rate"] = volume_pattern.current_rate
        
        # Update semantic baselines
        semantic_pattern = self._analyze_semantic_patterns()
        self.baseline_patterns["concepts"] = semantic_pattern.dominant_concepts
        self.baseline_patterns["domains"] = list(semantic_pattern.entity_type_distribution.keys())
        
        # Update diversity baseline
        metrics = self._calculate_stream_metrics()
        self.baseline_patterns["semantic_diversity"] = metrics.semantic_diversity
        
        # Persist to database
        await self._persist_baselines()
    
    async def _persist_baselines(self):
        """Persist baseline patterns to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store volume baselines
        cursor.execute('''
            INSERT OR REPLACE INTO volume_baselines
            (collection_name, baseline_rate, last_updated, document_count, pattern_stability)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            "global",
            self.baseline_patterns.get("volume_rate", 0.0),
            datetime.now().isoformat(),
            len(self.document_window),
            0.8  # Stability score
        ))
        
        conn.commit()
        conn.close()
    
    async def generate_reorganization_plan(self, drift_signal: DriftSignal) -> ReorganizationPlan:
        """Generate detailed reorganization plan based on drift signal."""
        logger.info(f"📋 Generating reorganization plan for {drift_signal.drift_type.value}")
        
        if drift_signal.recommended_action == ReorganizationAction.CREATE_SUBCATEGORY:
            return await self._plan_subcategory_creation(drift_signal)
        elif drift_signal.recommended_action == ReorganizationAction.SPLIT_COLLECTION:
            return await self._plan_collection_split(drift_signal)
        elif drift_signal.recommended_action == ReorganizationAction.MERGE_COLLECTIONS:
            return await self._plan_collection_merge(drift_signal)
        else:
            return ReorganizationPlan(
                action=ReorganizationAction.NO_ACTION,
                target_collections=[],
                new_collection_specs=[],
                migration_strategy="none",
                confidence=0.0,
                estimated_impact="minimal",
                rollback_plan={}
            )
    
    async def _plan_subcategory_creation(self, drift_signal: DriftSignal) -> ReorganizationPlan:
        """Plan creation of subcategories based on drift analysis."""
        return ReorganizationPlan(
            action=ReorganizationAction.CREATE_SUBCATEGORY,
            target_collections=drift_signal.affected_collections,
            new_collection_specs=[
                {
                    "name": f"{collection}_specialized",
                    "parent": collection,
                    "criteria": "high_volume_similar_documents"
                }
                for collection in drift_signal.affected_collections
            ],
            migration_strategy="gradual_migration",
            confidence=drift_signal.confidence,
            estimated_impact=f"Create specialized subcategories for {len(drift_signal.affected_collections)} collections",
            rollback_plan={
                "action": "merge_back_to_parent",
                "retention_period": "7_days"
            }
        )
    
    async def _plan_collection_split(self, drift_signal: DriftSignal) -> ReorganizationPlan:
        """Plan splitting of collections based on drift analysis."""
        return ReorganizationPlan(
            action=ReorganizationAction.SPLIT_COLLECTION,
            target_collections=drift_signal.affected_collections,
            new_collection_specs=[],
            migration_strategy="semantic_clustering",
            confidence=drift_signal.confidence,
            estimated_impact="Split overgrown collections into semantic clusters",
            rollback_plan={"action": "merge_splits", "window": "3_days"}
        )
    
    async def _plan_collection_merge(self, drift_signal: DriftSignal) -> ReorganizationPlan:
        """Plan merging of collections based on drift analysis."""
        return ReorganizationPlan(
            action=ReorganizationAction.MERGE_COLLECTIONS,
            target_collections=drift_signal.affected_collections,
            new_collection_specs=[],
            migration_strategy="similarity_based_merge",
            confidence=drift_signal.confidence,
            estimated_impact="Merge underutilized collections",
            rollback_plan={"action": "restore_splits", "backup_retention": "14_days"}
        )

# Global instance
concept_drift_detector = ConceptDriftDetector()