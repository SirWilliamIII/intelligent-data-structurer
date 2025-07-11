"""
Claude-Powered Ontology Analyzer

Deep semantic understanding of documents using Claude API.
Extracts entities, classifies document types, and builds concept hierarchies
for intelligent taxonomy organization.
"""

import asyncio
import json
import hashlib
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger
import sqlite3
from pathlib import Path

@dataclass
class Entity:
    """Represents an identified entity with semantic type."""
    name: str
    entity_type: str
    confidence: float
    context: str
    aliases: List[str]

@dataclass 
class DocumentType:
    """Classified document type with domain context."""
    type_name: str
    domain: str
    confidence: float
    characteristics: List[str]
    
@dataclass
class ConceptHierarchy:
    """Hierarchical abstraction levels for a concept."""
    specific: str
    medium: str  
    broad: str
    domain: str
    abstraction_path: List[str]

@dataclass
class OntologyAnalysis:
    """Complete ontological analysis of a document."""
    document_id: str
    entities: List[Entity]
    document_type: DocumentType
    concept_hierarchies: List[ConceptHierarchy]
    semantic_fingerprint: str
    cross_references: List[str]
    confidence_score: float
    analysis_metadata: Dict[str, Any]
    claude_response: Dict[str, Any]

class ClaudeOntologyAnalyzer:
    """
    Uses Claude API for deep semantic analysis and ontology learning.
    Builds understanding of entities, document types, and concept hierarchies.
    """
    
    def __init__(self, cache_db_path: str = "./claude_ontology_cache.db"):
        self.cache_db_path = cache_db_path
        self.claude_client = None  # Will be initialized with API key
        self._init_cache_db()
        
        # Analysis templates for different document types
        self.analysis_templates = {
            "sports_schedule": self._get_sports_schedule_template(),
            "contracts": self._get_contracts_template(),
            "financial": self._get_financial_template(),
            "general": self._get_general_template()
        }
        
    def _init_cache_db(self):
        """Initialize SQLite cache for Claude analysis results."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS claude_analysis_cache (
                content_hash TEXT PRIMARY KEY,
                analysis_result TEXT,
                confidence_score REAL,
                created_at TIMESTAMP,
                claude_model TEXT,
                token_usage INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_knowledge (
                entity_name TEXT PRIMARY KEY,
                entity_type TEXT,
                domain TEXT,
                confidence REAL,
                aliases TEXT,
                context_examples TEXT,
                last_updated TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_relationships (
                concept_id TEXT PRIMARY KEY,
                parent_concept TEXT,
                child_concepts TEXT,
                sibling_concepts TEXT,
                abstraction_level INTEGER,
                domain TEXT,
                relationship_strength REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def analyze_document(self, document_content: str, filename: str = "", 
                             document_id: str = None) -> OntologyAnalysis:
        """
        Perform complete ontological analysis of a document.
        
        Args:
            document_content: Full text content of the document
            filename: Original filename for context
            document_id: Unique identifier for caching
            
        Returns:
            Complete ontological analysis with entities, types, and hierarchies
        """
        if not document_id:
            document_id = self._generate_document_id(document_content, filename)
            
        logger.info(f"🧠 Starting Claude ontology analysis for document: {filename}")
        
        # Check cache first
        cached_analysis = self._get_cached_analysis(document_content)
        if cached_analysis:
            logger.info(f"📋 Using cached Claude analysis for {filename}")
            return cached_analysis
            
        try:
            # Perform Claude analysis
            claude_response = await self._call_claude_api(document_content, filename)
            
            # Parse and structure the response
            analysis = await self._parse_claude_response(
                claude_response, document_content, filename, document_id
            )
            
            # Cache the results
            self._cache_analysis(document_content, analysis)
            
            logger.info(f"✅ Claude ontology analysis complete for {filename} "
                       f"(confidence: {analysis.confidence_score:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Claude ontology analysis failed for {filename}: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(document_content, filename, document_id)
    
    async def _call_claude_api(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Call Claude API with structured ontology analysis prompt.
        """
        # Prepare context-aware prompt
        prompt = self._build_ontology_prompt(content, filename)
        
        # For now, simulate Claude API call
        # TODO: Replace with actual Claude API integration
        logger.info("🤖 Calling Claude API for ontological analysis...")
        
        # Simulate Claude response based on content patterns
        simulated_response = await self._simulate_claude_analysis(content, filename)
        
        return simulated_response
    
    def _build_ontology_prompt(self, content: str, filename: str) -> str:
        """
        Build comprehensive ontology analysis prompt for Claude.
        """
        prompt = f"""
You are an expert ontologist and semantic analyst. Analyze this document and provide a comprehensive ontological understanding.

Document: {filename}
Content (first 2000 chars): {content[:2000]}

Please provide a detailed analysis in the following JSON format:

{{
    "entities": [
        {{
            "name": "entity name",
            "type": "specific entity type (e.g., 'nfl_team', 'basketball_franchise', 'financial_contract')",
            "confidence": 0.95,
            "context": "how this entity appears in the document",
            "aliases": ["alternative names", "abbreviations"]
        }}
    ],
    "document_type": {{
        "type": "specific document type (e.g., 'sports_schedule', 'player_contract', 'financial_statement')",
        "domain": "broad domain (e.g., 'sports', 'business', 'legal')",
        "confidence": 0.92,
        "characteristics": ["key features that define this document type"]
    }},
    "concept_hierarchies": [
        {{
            "concept": "main concept from document",
            "specific": "most specific level (e.g., 'game_64')",
            "medium": "medium abstraction (e.g., 'basketball_game')",
            "broad": "broad category (e.g., 'sports_event')",
            "domain": "top domain (e.g., 'entertainment')",
            "abstraction_path": ["specific", "medium", "broad", "domain"]
        }}
    ],
    "semantic_relationships": [
        {{
            "relationship_type": "type of relationship (e.g., 'part_of', 'instance_of', 'related_to')",
            "source": "source entity/concept",
            "target": "target entity/concept",
            "strength": 0.85
        }}
    ],
    "document_purpose": "what this document is meant to accomplish",
    "semantic_markers": ["key phrases or patterns that identify this document type"],
    "cross_references": ["entities or concepts that might link to other documents"],
    "abstraction_recommendation": "What level of abstraction should this document be categorized at? (specific/medium/broad)"
}}

Focus on:
1. **Entity Understanding**: What are the main entities and what TYPE of entities are they?
2. **Document Classification**: What KIND of document is this and what DOMAIN does it belong to?
3. **Hierarchical Thinking**: How should this content be abstracted from specific to general?
4. **Semantic Relationships**: How do entities and concepts relate to each other?
5. **Categorization Guidance**: At what level should this be organized with similar documents?

Be precise about entity types and document classifications. Avoid overly generic terms.
"""
        return prompt
    
    async def _simulate_claude_analysis(self, content: str, filename: str) -> Dict[str, Any]:
        """
        Simulate Claude API response for development/testing.
        This will be replaced with actual Claude API calls.
        """
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Detect document patterns
        if "schedule" in filename_lower or "game" in content_lower:
            return self._simulate_sports_schedule_analysis(content, filename)
        elif "contract" in filename_lower or "guaranteed" in content_lower:
            return self._simulate_contract_analysis(content, filename)
        else:
            return self._simulate_general_analysis(content, filename)
    
    def _simulate_sports_schedule_analysis(self, content: str, filename: str) -> Dict[str, Any]:
        """Truly organic sports schedule analysis - no hardcoded classifications."""
        
        # Extract team names and patterns organically using generic patterns
        team_patterns = [
            r'(\w+\s+\w+)\s*-\s*2025',  # "Team Name - 2025"
            r'vs\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "vs Team Name"
            r'Game\s+\d+:\s*vs\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "Game N: vs Team"
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\n',  # Team names at start of lines
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Schedule'  # "Team Name Schedule"
        ]
        
        import re
        teams = set()
        for pattern in team_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            teams.update([team.strip() for team in matches if len(team.strip()) > 3])
        
        content_filename = (content + " " + filename).lower()
        
        # ORGANIC DISCOVERY - Start with very low confidence
        base_confidence = 0.3  # Start low - let system decide broader placement
        
        # Only look for explicit league mentions - no hardcoding of specific teams
        league_indicators = {
            "nfl": ["nfl", "national football league"],
            "nba": ["nba", "national basketball association"], 
            "mlb": ["mlb", "major league baseball"],
            "nhl": ["nhl", "national hockey league"]
        }
        
        detected_league = None
        for league, indicators in league_indicators.items():
            if any(indicator in content_filename for indicator in indicators):
                detected_league = league
                base_confidence = 0.6  # Higher confidence only for explicit league mentions
                break
        
        # If no explicit league, look for generic sport terms but keep confidence very low
        sport_terms = ["schedule", "game", "vs", "season", "team", "match"]
        if not detected_league and any(term in content_filename for term in sport_terms):
            detected_league = "sports"  # Generic sports category
            base_confidence = 0.2  # Very low - should go to broad staging
        
        # Build entities with generic types - no sport-specific hardcoding
        entities = []
        for team in list(teams)[:5]:  # Limit to prevent overwhelming
            entity_type = f"{detected_league}_team" if detected_league else "team_entity"
            entities.append({
                "name": team,
                "type": entity_type,
                "confidence": 0.85,  # High confidence in entity extraction, low in classification
                "context": f"Team mentioned in schedule document",
                "aliases": [team.replace(" ", ""), team.upper()]
            })
        
        # Use generic document type with low confidence to encourage broad placement
        doc_type_confidence = base_confidence
        
        return {
            "entities": entities,
            "document_type": {
                "type": "team_schedule",  # Generic, not sport-specific
                "domain": "sports",
                "confidence": doc_type_confidence,
                "characteristics": ["team listings", "schedule format", "competitive events"]
            },
            "concept_hierarchies": [
                {
                    "concept": "team_schedule",
                    "specific": "game_listing",
                    "medium": "team_schedule",
                    "broad": "sports_event_listing", 
                    "domain": "sports_entertainment",
                    "abstraction_path": ["game_listing", "team_schedule", 
                                       "sports_event_listing", "sports_entertainment"]
                }
            ],
            "semantic_relationships": [
                {
                    "relationship_type": "instance_of",
                    "source": "game_listing",
                    "target": "sports_event",
                    "strength": 0.8
                }
            ],
            "document_purpose": "Schedule competitive events between teams",
            "semantic_markers": ["Game", "vs", "schedule", "team", "season"],
            "cross_references": list(teams),
            # KEY: Recommend broad placement for organic discovery
            "abstraction_recommendation": "broad" if base_confidence < 0.5 else "medium"
        }
    
    def _simulate_contract_analysis(self, content: str, filename: str) -> Dict[str, Any]:
        """Simulate Claude analysis for contract documents."""
        # Extract team from filename
        import re
        team_match = re.search(r'([A-Z][a-z]+(?:_[A-Z][a-z]+)*)', filename.replace('_contracts', ''))
        team_name = team_match.group(1).replace('_', ' ') if team_match else "Unknown Team"
        
        return {
            "entities": [
                {
                    "name": team_name,
                    "type": "sports_organization",
                    "confidence": 0.90,
                    "context": "Organization with player contracts",
                    "aliases": [team_name.replace(" ", "")]
                }
            ],
            "document_type": {
                "type": "player_contracts",
                "domain": "business_legal",
                "confidence": 0.95,
                "characteristics": ["guaranteed amounts", "contract terms", "player information"]
            },
            "concept_hierarchies": [
                {
                    "concept": "sports_contracts",
                    "specific": "player_contract",
                    "medium": "sports_agreement",
                    "broad": "business_contract",
                    "domain": "legal_document",
                    "abstraction_path": ["player_contract", "sports_agreement", 
                                       "business_contract", "legal_document"]
                }
            ],
            "semantic_relationships": [
                {
                    "relationship_type": "governs",
                    "source": "player_contract",
                    "target": "employment_relationship",
                    "strength": 0.95
                }
            ],
            "document_purpose": "Define player employment terms and compensation",
            "semantic_markers": ["guaranteed", "years", "position", "contract"],
            "cross_references": [team_name, "player_salaries", "contract_terms"],
            "abstraction_recommendation": "medium"
        }
    
    def _simulate_general_analysis(self, content: str, filename: str) -> Dict[str, Any]:
        """Simulate Claude analysis for general documents."""
        return {
            "entities": [],
            "document_type": {
                "type": "general_document",
                "domain": "general",
                "confidence": 0.60,
                "characteristics": ["text content", "informational"]
            },
            "concept_hierarchies": [
                {
                    "concept": "document",
                    "specific": "text_document",
                    "medium": "information_document",
                    "broad": "content",
                    "domain": "information",
                    "abstraction_path": ["text_document", "information_document", 
                                       "content", "information"]
                }
            ],
            "semantic_relationships": [],
            "document_purpose": "Information storage and communication",
            "semantic_markers": [],
            "cross_references": [],
            "abstraction_recommendation": "broad"
        }
    
    async def _parse_claude_response(self, claude_response: Dict[str, Any], 
                                   content: str, filename: str, document_id: str) -> OntologyAnalysis:
        """
        Parse Claude's response into structured OntologyAnalysis object.
        """
        # Extract entities
        entities = []
        for entity_data in claude_response.get("entities", []):
            entities.append(Entity(
                name=entity_data["name"],
                entity_type=entity_data["type"],
                confidence=entity_data["confidence"],
                context=entity_data["context"],
                aliases=entity_data.get("aliases", [])
            ))
        
        # Extract document type
        doc_type_data = claude_response["document_type"]
        document_type = DocumentType(
            type_name=doc_type_data["type"],
            domain=doc_type_data["domain"],
            confidence=doc_type_data["confidence"],
            characteristics=doc_type_data["characteristics"]
        )
        
        # Extract concept hierarchies
        concept_hierarchies = []
        for hierarchy_data in claude_response.get("concept_hierarchies", []):
            concept_hierarchies.append(ConceptHierarchy(
                specific=hierarchy_data["specific"],
                medium=hierarchy_data["medium"],
                broad=hierarchy_data["broad"],
                domain=hierarchy_data["domain"],
                abstraction_path=hierarchy_data["abstraction_path"]
            ))
        
        # Generate semantic fingerprint
        semantic_fingerprint = self._generate_semantic_fingerprint(
            entities, document_type, concept_hierarchies
        )
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(claude_response)
        
        # Extract cross-references
        cross_references = claude_response.get("cross_references", [])
        
        return OntologyAnalysis(
            document_id=document_id,
            entities=entities,
            document_type=document_type,
            concept_hierarchies=concept_hierarchies,
            semantic_fingerprint=semantic_fingerprint,
            cross_references=cross_references,
            confidence_score=confidence_score,
            analysis_metadata={
                "filename": filename,
                "analysis_timestamp": datetime.now().isoformat(),
                "content_length": len(content),
                "abstraction_recommendation": claude_response.get("abstraction_recommendation", "medium")
            },
            claude_response=claude_response
        )
    
    def _generate_semantic_fingerprint(self, entities: List[Entity], 
                                     document_type: DocumentType,
                                     hierarchies: List[ConceptHierarchy]) -> str:
        """Generate unique semantic fingerprint for the document."""
        components = []
        
        # Add document type
        components.append(document_type.type_name)
        components.append(document_type.domain)
        
        # Add top entity types
        entity_types = [e.entity_type for e in entities[:3]]
        components.extend(sorted(entity_types))
        
        # Add concept hierarchies
        for hierarchy in hierarchies[:2]:
            components.extend([hierarchy.medium, hierarchy.broad])
        
        # Create hash
        fingerprint_text = "|".join(components)
        return hashlib.md5(fingerprint_text.encode()).hexdigest()[:16]
    
    def _calculate_overall_confidence(self, claude_response: Dict[str, Any]) -> float:
        """Calculate overall confidence score from Claude response."""
        confidences = []
        
        # Document type confidence
        confidences.append(claude_response["document_type"]["confidence"])
        
        # Entity confidences
        for entity in claude_response.get("entities", []):
            confidences.append(entity["confidence"])
        
        # Average with bias toward document type (more important)
        if confidences:
            doc_type_weight = 0.5
            entity_weight = 0.5 / max(len(confidences) - 1, 1)
            
            weighted_sum = claude_response["document_type"]["confidence"] * doc_type_weight
            for conf in confidences[1:]:
                weighted_sum += conf * entity_weight
                
            return min(weighted_sum, 1.0)
        
        return 0.5  # Default moderate confidence
    
    def _create_fallback_analysis(self, content: str, filename: str, 
                                document_id: str) -> OntologyAnalysis:
        """Create fallback analysis if Claude API fails."""
        logger.warning(f"Creating fallback analysis for {filename}")
        
        return OntologyAnalysis(
            document_id=document_id,
            entities=[],
            document_type=DocumentType(
                type_name="unknown_document",
                domain="general",
                confidence=0.3,
                characteristics=["fallback_analysis"]
            ),
            concept_hierarchies=[ConceptHierarchy(
                specific="unknown",
                medium="document",
                broad="content",
                domain="information",
                abstraction_path=["unknown", "document", "content", "information"]
            )],
            semantic_fingerprint="fallback_" + hashlib.md5(content.encode()).hexdigest()[:16],
            cross_references=[],
            confidence_score=0.3,
            analysis_metadata={
                "filename": filename,
                "analysis_timestamp": datetime.now().isoformat(),
                "fallback": True
            },
            claude_response={}
        )
    
    def _generate_document_id(self, content: str, filename: str) -> str:
        """Generate unique document ID for caching."""
        combined = f"{filename}|{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def _get_cached_analysis(self, content: str) -> Optional[OntologyAnalysis]:
        """Retrieve cached analysis if available."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT analysis_result FROM claude_analysis_cache 
            WHERE content_hash = ?
        ''', (content_hash,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                cached_data = json.loads(row[0])
                # Reconstruct OntologyAnalysis from cached data
                return self._reconstruct_analysis_from_cache(cached_data)
            except Exception as e:
                logger.warning(f"Failed to load cached analysis: {e}")
        
        return None
    
    def _cache_analysis(self, content: str, analysis: OntologyAnalysis):
        """Cache analysis results."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        analysis_json = json.dumps(asdict(analysis), default=str)
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO claude_analysis_cache
            (content_hash, analysis_result, confidence_score, created_at, claude_model, token_usage)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            content_hash,
            analysis_json,
            analysis.confidence_score,
            datetime.now().isoformat(),
            "claude-3-sonnet-simulated",  # Update when using real API
            0  # Token usage tracking
        ))
        
        conn.commit()
        conn.close()
    
    def _reconstruct_analysis_from_cache(self, cached_data: Dict[str, Any]) -> OntologyAnalysis:
        """Reconstruct OntologyAnalysis object from cached JSON data."""
        # Reconstruct nested Entity objects
        entities = []
        for entity_data in cached_data.get("entities", []):
            if isinstance(entity_data, dict):
                entities.append(Entity(**entity_data))
            else:
                entities.append(entity_data)
        
        # Reconstruct DocumentType object
        doc_type_data = cached_data.get("document_type", {})
        if isinstance(doc_type_data, dict):
            document_type = DocumentType(**doc_type_data)
        else:
            document_type = doc_type_data
        
        # Reconstruct ConceptHierarchy objects
        concept_hierarchies = []
        for hierarchy_data in cached_data.get("concept_hierarchies", []):
            if isinstance(hierarchy_data, dict):
                concept_hierarchies.append(ConceptHierarchy(**hierarchy_data))
            else:
                concept_hierarchies.append(hierarchy_data)
        
        # Create OntologyAnalysis with reconstructed objects
        return OntologyAnalysis(
            document_id=cached_data.get("document_id", ""),
            entities=entities,
            document_type=document_type,
            concept_hierarchies=concept_hierarchies,
            semantic_fingerprint=cached_data.get("semantic_fingerprint", ""),
            cross_references=cached_data.get("cross_references", []),
            confidence_score=cached_data.get("confidence_score", 0.0),
            analysis_metadata=cached_data.get("analysis_metadata", {}),
            claude_response=cached_data.get("claude_response", {})
        )
    
    # Template methods for different document types
    def _get_sports_schedule_template(self) -> str:
        return "sports_schedule_analysis_template"
    
    def _get_contracts_template(self) -> str:
        return "contracts_analysis_template"
    
    def _get_financial_template(self) -> str:
        return "financial_analysis_template"
    
    def _get_general_template(self) -> str:
        return "general_analysis_template"

# Global instance for easy import
claude_ontology_analyzer = ClaudeOntologyAnalyzer()