"""
Intelligent content analyzer that learns and evolves table structures.
Uses deep semantic understanding to create truly adaptive database schemas.
"""

import re
import json
import spacy
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger
from collections import Counter, defaultdict
import sqlite3
from pathlib import Path
from .classifier import IntelligentClassifier, ContentType
from .business_classifier import BusinessDocumentClassifier, BusinessClassificationResult
from .semantic_enrichment import semantic_enricher

@dataclass
class ContentSignature:
    """Semantic signature of content for similarity matching."""
    domain_keywords: Set[str]
    structural_patterns: Set[str]
    entity_types: Set[str]
    content_markers: Set[str]
    semantic_hash: str

@dataclass
class TableEvolution:
    """Tracks how a table schema should evolve."""
    table_name: str
    current_schema: Dict[str, str]
    suggested_additions: Dict[str, str]
    pattern_frequency: Dict[str, int]
    confidence_score: float
    sample_contents: List[str]

@dataclass
class IntelligentResult:
    """Result of intelligent content analysis."""
    table_name: str
    confidence: float
    reasoning: str
    schema_evolution: TableEvolution
    extracted_data: Dict[str, Any]
    similar_content_ids: List[str]
    learning_feedback: Dict[str, Any]

class ContentLearningSystem:
    """System that learns from content patterns and evolves schemas."""
    
    def __init__(self, learning_db_path: str = "./content_learning.db"):
        self.learning_db_path = learning_db_path
        self._init_learning_db()
        self.content_signatures = {}
        self.table_patterns = defaultdict(list)
        self.schema_evolution = {}
        
    def _init_learning_db(self):
        """Initialize the learning database."""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Content signatures table - stores semantic fingerprints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_signatures (
                content_id TEXT PRIMARY KEY,
                semantic_hash TEXT,
                domain_keywords TEXT,
                structural_patterns TEXT,
                entity_types TEXT,
                table_assigned TEXT,
                confidence REAL,
                created_at TIMESTAMP
            )
        ''')
        
        # Schema evolution table - tracks how schemas change
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_evolution (
                table_name TEXT,
                column_name TEXT,
                column_type TEXT,
                frequency INTEGER,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                PRIMARY KEY (table_name, column_name)
            )
        ''')
        
        # Pattern learning table - learns from content patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_learning (
                pattern_hash TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_content TEXT,
                frequency INTEGER,
                associated_tables TEXT,
                confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def analyze_content_signature(self, content: str, entities: List[Dict]) -> ContentSignature:
        """Create a semantic signature for content similarity matching using dynamic learning."""
        
        # Extract meaningful words (nouns, verbs, adjectives) dynamically
        domain_keywords = set()
        content_lower = content.lower()
        
        # Extract significant terms dynamically from content
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content_lower)
        word_freq = Counter(words)
        
        # Get top meaningful words (excluding common stop words)
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        for word, freq in word_freq.most_common(20):
            if word not in stop_words and freq > 1:
                domain_keywords.add(word)
        
        # Add entity-based keywords (but filter out common person names that might be sport names)
        for entity in entities:
            # Only add entities that are clearly organizational or technical
            if entity['label'] in ['ORG', 'PRODUCT', 'TECHNOLOGY', 'NORP', 'EVENT']:
                domain_keywords.add(entity['text'].lower())
        
        # ORGANIC SEMANTIC ENRICHMENT - Discover what entities have in common
        entity_names = [entity['text'] for entity in entities if entity['label'] == 'ORG']
        if len(entity_names) >= 2:
            try:
                enriched_keywords = await semantic_enricher.enrich_entities(
                    entities=entity_names,
                    content_context=content[:500]  # Provide context for better search
                )
                
                # Add discovered domain keywords
                for entity_name, keywords in enriched_keywords.items():
                    domain_keywords.update(keywords)
                    logger.info(f"Enriched '{entity_name}' with keywords: {keywords}")
                    
            except Exception as e:
                logger.warning(f"Semantic enrichment failed: {e}")
                # Continue with original keywords if enrichment fails
        
        # Detect structural patterns dynamically
        structural_patterns = set()
        
        # List structures
        if re.search(r'\n\s*[-*•]\s+', content) or re.search(r'\n\s*\d+\.\s+', content):
            structural_patterns.add('list_format')
        
        # Simple list (lines with single words/phrases)
        lines = content.split('\n')
        short_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) < 50]
        if len(short_lines) > 5 and len(short_lines) / len(lines) > 0.5:
            structural_patterns.add('simple_list')
        
        # Command structures
        if re.search(r'\$\s+\w+', content) or re.search(r'```', content):
            structural_patterns.add('command_format')
        
        # Schedule structures
        if re.search(r'\d{1,2}:\d{2}', content) and re.search(r'\d{1,2}[/-]\d{1,2}', content):
            structural_patterns.add('schedule_format')
        
        # Recipe structures
        if re.search(r'\d+\s*(?:cups?|tbsp|tsp)', content, re.IGNORECASE):
            structural_patterns.add('recipe_format')
        
        # Reference structures
        if content.count(':') > 5 and len(content.split('\n')) > 10:
            structural_patterns.add('reference_format')
        
        # Entity types (but be more selective)
        entity_types = set()
        for entity in entities:
            # Skip PERSON entities for lists that might be sports/games
            if entity['label'] != 'PERSON' or len(entities) < 10:
                entity_types.add(entity['label'])
        
        # Content markers (specific identifiers) - keep minimal and specific
        content_markers = set()
        
        # Only add very specific technical markers
        if 'kubernetes' in content_lower or 'kubectl' in content_lower:
            content_markers.add('kubernetes')
        if 'docker' in content_lower and 'container' in content_lower:
            content_markers.add('docker')
        if 'git' in content_lower and ('commit' in content_lower or 'repository' in content_lower):
            content_markers.add('git')
        
        # Create semantic hash
        signature_data = {
            'domain_keywords': sorted(domain_keywords),
            'structural_patterns': sorted(structural_patterns),
            'entity_types': sorted(entity_types),
            'content_markers': sorted(content_markers)
        }
        
        semantic_hash = hashlib.md5(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        
        return ContentSignature(
            domain_keywords=domain_keywords,
            structural_patterns=structural_patterns,
            entity_types=entity_types,
            content_markers=content_markers,
            semantic_hash=semantic_hash
        )
    
    def find_similar_content(self, signature: ContentSignature, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find content with similar semantic signatures."""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT content_id, domain_keywords, structural_patterns, entity_types FROM content_signatures')
        results = cursor.fetchall()
        conn.close()
        
        similar_content = []
        
        for content_id, domain_kw_str, struct_pat_str, entity_types_str in results:
            stored_domain = set(json.loads(domain_kw_str))
            stored_structural = set(json.loads(struct_pat_str))
            stored_entities = set(json.loads(entity_types_str))
            
            # Calculate similarity scores
            domain_similarity = len(signature.domain_keywords & stored_domain) / max(len(signature.domain_keywords | stored_domain), 1)
            structural_similarity = len(signature.structural_patterns & stored_structural) / max(len(signature.structural_patterns | stored_structural), 1)
            entity_similarity = len(signature.entity_types & stored_entities) / max(len(signature.entity_types | stored_entities), 1)
            
            # Weighted overall similarity
            overall_similarity = (domain_similarity * 0.4 + structural_similarity * 0.4 + entity_similarity * 0.2)
            
            if overall_similarity >= threshold:
                similar_content.append((content_id, overall_similarity))
        
        return sorted(similar_content, key=lambda x: x[1], reverse=True)
    
    def learn_from_content(self, content_id: str, signature: ContentSignature, table_name: str, confidence: float):
        """Learn from content and update knowledge base."""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Store content signature
        cursor.execute('''
            INSERT OR REPLACE INTO content_signatures 
            (content_id, semantic_hash, domain_keywords, structural_patterns, entity_types, table_assigned, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content_id,
            signature.semantic_hash,
            json.dumps(list(signature.domain_keywords)),
            json.dumps(list(signature.structural_patterns)),
            json.dumps(list(signature.entity_types)),
            table_name,
            confidence,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def suggest_table_evolution(self, table_name: str, content_patterns: List[Dict]) -> TableEvolution:
        """Suggest how a table schema should evolve based on content patterns."""
        
        # Analyze patterns in content assigned to this table
        pattern_frequency = defaultdict(int)
        suggested_columns = {}
        
        for pattern in content_patterns:
            # Count frequency of different data types found
            if pattern.get('emails'):
                pattern_frequency['has_emails'] += 1
            if pattern.get('phones'):
                pattern_frequency['has_phones'] += 1
            if pattern.get('dates'):
                pattern_frequency['has_dates'] += 1
            if pattern.get('commands'):
                pattern_frequency['has_commands'] += 1
            if pattern.get('ingredients'):
                pattern_frequency['has_ingredients'] += 1
            if pattern.get('teams'):
                pattern_frequency['has_teams'] += 1
        
        total_content = len(content_patterns)
        
        # Suggest columns if they appear in >50% of content
        if pattern_frequency['has_emails'] / total_content > 0.5:
            suggested_columns['email_addresses'] = 'JSONB'
        
        if pattern_frequency['has_commands'] / total_content > 0.5:
            suggested_columns['commands'] = 'JSONB'
        
        if pattern_frequency['has_ingredients'] / total_content > 0.5:
            suggested_columns['ingredients'] = 'JSONB'
            suggested_columns['prep_instructions'] = 'JSONB'
        
        if pattern_frequency['has_teams'] / total_content > 0.5:
            suggested_columns['teams_involved'] = 'JSONB'
            suggested_columns['event_details'] = 'JSONB'
        
        # Calculate confidence based on pattern consistency
        confidence = sum(pattern_frequency.values()) / (total_content * len(pattern_frequency))
        
        return TableEvolution(
            table_name=table_name,
            current_schema={},  # Would be populated from actual DB schema
            suggested_additions=suggested_columns,
            pattern_frequency=dict(pattern_frequency),
            confidence_score=confidence,
            sample_contents=[p.get('content', '')[:200] for p in content_patterns[:3]]
        )

class IntelligentAnalyzer:
    """Intelligent analyzer that learns and evolves understanding."""
    
    def __init__(self):
        self.nlp = None
        self._load_models()
        self.learning_system = ContentLearningSystem()
        self.classifier = IntelligentClassifier()
        self.business_classifier = BusinessDocumentClassifier()
        
        # Dynamic table naming strategies
        self.naming_strategies = {
            'kubernetes': lambda: 'kubernetes_resources',
            'docker': lambda: 'docker_references', 
            'football': lambda: 'football_schedules',
            'recipe': lambda: 'cooking_recipes',
            'schedule': lambda: 'event_schedules',
            'cheat_sheet': lambda: 'reference_guides'
        }
    
    def _load_models(self):
        """Load NLP models."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Intelligent NLP models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            self.nlp = None
    
    async def analyze_content(self, content: str, filename: str) -> IntelligentResult:
        """Intelligently analyze content with learning and evolution."""
        
        if not self.nlp:
            return self._fallback_analysis(content, filename)
        
        # Generate content ID
        content_id = hashlib.md5(f"{filename}_{content[:100]}".encode()).hexdigest()
        
        # Process with spaCy
        doc = self.nlp(content)
        entities = [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) or ent.label_
            }
            for ent in doc.ents
        ]
        
        # Create content signature
        signature = await self.learning_system.analyze_content_signature(content, entities)
        
        # Use business classifier first, then fallback to general classifier
        business_result = await self.business_classifier.classify_business_document(content, filename)
        
        # If business classification is confident, use it; otherwise use general classifier
        if business_result.confidence > 0.5:
            classification_result = None  # Use business result directly
        else:
            classification_result = await self.classifier.classify_content(content, filename)
        
        # Find similar content and check for duplicates
        similar_content = self.learning_system.find_similar_content(signature)
        
        # Check for near-duplicate content (>0.95 similarity)
        if similar_content and similar_content[0][1] > 0.95:
            logger.warning(f"Near-duplicate content detected for {filename} (similarity: {similar_content[0][1]:.1%})")
            # Could return existing document ID instead of processing
        
        # Determine table name intelligently (enhanced with classification)
        table_name, confidence, reasoning = self._determine_intelligent_table(
            signature, similar_content, content, filename, classification_result, business_result
        )
        
        # Extract comprehensive data
        extracted_data = self._extract_intelligent_data(content, entities, signature, filename, classification_result, business_result)
        
        # Add content_type and domain fields that are missing
        if business_result.confidence > 0.5:
            extracted_data['content_type'] = business_result.business_type.value if business_result.business_type else 'unknown'
            extracted_data['classification_confidence'] = business_result.confidence
            extracted_data['classification_reasoning'] = business_result.reasoning
            extracted_data['business_category'] = business_result.category
            extracted_data['business_metadata'] = business_result.metadata
        elif classification_result:
            extracted_data['content_type'] = classification_result.content_type.value
            extracted_data['classification_confidence'] = classification_result.confidence
            extracted_data['classification_reasoning'] = classification_result.reasoning
        else:
            extracted_data['content_type'] = self._map_table_to_content_type(table_name)
        extracted_data['domain'] = self._extract_domain_from_table(table_name)
        
        # Suggest schema evolution
        content_patterns = [extracted_data]  # In real implementation, would include similar content
        schema_evolution = self.learning_system.suggest_table_evolution(table_name, content_patterns)
        
        # Learn from this content
        self.learning_system.learn_from_content(content_id, signature, table_name, confidence)
        
        # Generate learning feedback
        learning_feedback = self._generate_learning_feedback(signature, similar_content, confidence)
        
        return IntelligentResult(
            table_name=table_name,
            confidence=confidence,
            reasoning=reasoning,
            schema_evolution=schema_evolution,
            extracted_data=extracted_data,
            similar_content_ids=[content_id for content_id, _ in similar_content],
            learning_feedback=learning_feedback
        )
    
    def _determine_intelligent_table(self, signature: ContentSignature, similar_content: List[Tuple[str, float]], 
                                   content: str, filename: str, classification_result=None, business_result=None) -> Tuple[str, float, str]:
        """Intelligently determine table name using learning and similarity."""
        
        reasoning_parts = []
        
        # If we have confident business classification, use it (highest priority)
        if business_result and business_result.confidence > 0.5:
            table_name = business_result.suggested_collection
            confidence = business_result.confidence
            reasoning_parts.append(f"Business document type: {business_result.business_type.value if business_result.business_type else 'unknown'}")
            reasoning_parts.append(f"Category: {business_result.category}")
            return table_name, confidence, "; ".join(reasoning_parts)
        
        # If we have very similar content, use the same table (second priority)
        if similar_content and similar_content[0][1] > 0.7:
            # Get table name from most similar content
            conn = sqlite3.connect(self.learning_system.learning_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT table_assigned FROM content_signatures WHERE content_id = ?', 
                          (similar_content[0][0],))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                table_name = result[0]
                confidence = similar_content[0][1]
                reasoning_parts.append(f"Matched similar content with {confidence:.1%} similarity")
                return table_name, confidence, "; ".join(reasoning_parts)
        
        # Generate collection name dynamically based on content patterns
        table_name = self._generate_dynamic_collection_name(signature, content, filename)
        
        # Calculate confidence based on how distinctive the content is
        confidence = self._calculate_collection_confidence(signature, content, filename)
        
        reasoning_parts.append(f"Dynamic collection '{table_name}' created based on content patterns")
        reasoning_parts.append(f"Top keywords: {', '.join(list(signature.domain_keywords)[:3])}")
        reasoning_parts.append(f"Structure: {', '.join(signature.structural_patterns)}")
        
        return table_name, confidence, "; ".join(reasoning_parts)
    
    def _generate_dynamic_collection_name(self, signature: ContentSignature, content: str, filename: str) -> str:
        """Generate a meaningful collection name based on content patterns."""
        
        # Use the most frequent meaningful keywords to create a collection name
        keywords = list(signature.domain_keywords)
        
        # Start with specific technical markers
        if 'kubernetes' in signature.content_markers:
            return 'kubernetes_resources'
        if 'docker' in signature.content_markers:
            return 'docker_references'
        if 'git' in signature.content_markers:
            return 'git_references'
        
        # Use structural patterns for generic types
        if 'simple_list' in signature.structural_patterns:
            # For simple lists, use the most common meaningful word
            if keywords:
                primary_keyword = keywords[0]
                # Create a collection name based on the primary keyword
                if len(keywords) == 1:
                    return f"{primary_keyword}_list"
                else:
                    # Multiple keywords - create a more general collection
                    return f"{primary_keyword}_collection"
            else:
                return 'simple_lists'
        
        if 'recipe_format' in signature.structural_patterns:
            return 'cooking_recipes'
        
        if 'command_format' in signature.structural_patterns:
            return 'technical_references'
        
        if 'schedule_format' in signature.structural_patterns:
            return 'event_schedules'
        
        # Domain-based naming using most frequent keywords
        if keywords:
            primary_keyword = keywords[0]
            
            # Check if it's a technical domain
            tech_indicators = ['command', 'server', 'api', 'database', 'config', 'install', 'error', 'system']
            if any(indicator in keywords for indicator in tech_indicators):
                return f"{primary_keyword}_technical"
            
            # Check if it's a sports domain
            sports_indicators = ['team', 'game', 'match', 'player', 'score', 'season', 'league', 'football', 'basketball']
            if any(indicator in keywords for indicator in sports_indicators):
                return f"{primary_keyword}_sports"
            
            # Check if it's a cooking domain
            cooking_indicators = ['recipe', 'ingredient', 'cook', 'bake', 'serve', 'tablespoon', 'cup']
            if any(indicator in keywords for indicator in cooking_indicators):
                return f"{primary_keyword}_cooking"
            
            # Generic domain-based naming
            return f"{primary_keyword}_documents"
        
        # Filename-based fallback
        filename_lower = filename.lower()
        if 'cheat' in filename_lower or 'reference' in filename_lower:
            return 'reference_documents'
        
        if 'schedule' in filename_lower or 'calendar' in filename_lower:
            return 'schedule_documents'
        
        # Default
        return 'general_documents'
    
    def _calculate_collection_confidence(self, signature: ContentSignature, content: str, filename: str) -> float:
        """Calculate confidence in collection assignment based on content distinctiveness."""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence for specific technical markers
        if signature.content_markers:
            confidence += 0.3
        
        # Boost confidence for clear structural patterns
        if signature.structural_patterns:
            confidence += 0.2
        
        # Boost confidence for rich domain keywords
        if len(signature.domain_keywords) > 3:
            confidence += 0.2
        
        # Boost confidence for simple lists (clear structure)
        if 'simple_list' in signature.structural_patterns:
            confidence += 0.1
        
        # Reduce confidence for very generic content
        if not signature.domain_keywords and not signature.structural_patterns:
            confidence -= 0.2
        
        return min(confidence, 1.0)
    
    def _extract_intelligent_data(self, content: str, entities: List[Dict], 
                                signature: ContentSignature, filename: str, classification_result=None, business_result=None) -> Dict[str, Any]:
        """Extract data with intelligence about content type."""
        
        # Base extraction
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        phones = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', content)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b', content)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        
        # Clean content by removing excessive whitespace while preserving structure
        cleaned_content = self._clean_content(content)
        
        data = {
            'content': cleaned_content,
            'entities': entities,
            'emails': emails,
            'phones': phones,
            'dates': dates,
            'urls': urls,
            'word_count': len(content.split()),
            'char_count': len(cleaned_content),
            'original_char_count': len(content),
            'source_file': filename,
            'extracted_at': datetime.utcnow().isoformat(),
            
            # Intelligent additions (convert sets to lists for JSON serialization)
            'semantic_signature': {
                'domain_keywords': list(signature.domain_keywords),
                'structural_patterns': list(signature.structural_patterns),
                'entity_types': list(signature.entity_types),
                'content_markers': list(signature.content_markers),
                'semantic_hash': signature.semantic_hash
            },
            'priority': self._calculate_intelligent_priority(content, signature),
            'content_markers': list(signature.content_markers),
            'structural_patterns': list(signature.structural_patterns)
        }
        
        # Store content-specific extractions in a separate field to avoid column issues
        content_specific = {}
        
        if 'kubernetes' in signature.content_markers:
            content_specific.update(self._extract_kubernetes_data(content))
        
        if 'recipe_format' in signature.structural_patterns:
            content_specific.update(self._extract_recipe_data(content))
        
        if 'sports_schedule' in signature.content_markers:
            content_specific.update(self._extract_sports_data(content))
        
        if 'command_format' in signature.structural_patterns:
            content_specific.update(self._extract_command_data(content))
        
        # Add classification-based extraction
        if classification_result:
            if classification_result.content_type == ContentType.CONTACT_INFO:
                content_specific.update(self._extract_contact_data(content))
            elif classification_result.content_type == ContentType.PRODUCT_DATA:
                content_specific.update(self._extract_product_data(content))
            elif classification_result.content_type == ContentType.EVENT_INFO:
                content_specific.update(self._extract_event_data(content))
            elif classification_result.content_type == ContentType.RECIPE:
                content_specific.update(self._extract_recipe_data(content))
        
        # Add content-specific data as JSON to avoid column mismatches
        if content_specific:
            data['content_specific_data'] = content_specific
        
        return data
    
    def _extract_kubernetes_data(self, content: str) -> Dict[str, Any]:
        """Extract Kubernetes-specific data."""
        namespaces = re.findall(r'namespace[:\s]+([a-z0-9-]+)', content, re.IGNORECASE)
        kubectl_commands = re.findall(r'kubectl\s+([^\n]+)', content)
        resources = re.findall(r'\b(pod|service|deployment|configmap|secret)[s]?\b', content, re.IGNORECASE)
        
        return {
            'namespaces': list(set(namespaces)),
            'kubectl_commands': kubectl_commands,
            'k8s_resources': list(set(resources)),
            'technology': 'kubernetes'
        }
    
    def _extract_recipe_data(self, content: str) -> Dict[str, Any]:
        """Extract recipe-specific data."""
        ingredients = re.findall(r'(\d+(?:\.\d+)?)\s*(cups?|tbsp|tsp|lbs?|oz|grams?)\s+([^\n,]+)', content, re.IGNORECASE)
        cook_times = re.findall(r'(\d+)\s*(minutes?|hours?|mins?|hrs?)', content, re.IGNORECASE)
        temperatures = re.findall(r'(\d+)\s*°?[Ff]', content)
        
        return {
            'ingredients_list': [{'amount': ing[0], 'unit': ing[1], 'item': ing[2]} for ing in ingredients],
            'cooking_times': [{'duration': time[0], 'unit': time[1]} for time in cook_times],
            'temperatures': [f"{temp}°F" for temp in temperatures],
            'content_type': 'recipe'
        }
    
    def _extract_sports_data(self, content: str) -> Dict[str, Any]:
        """Extract sports-specific data."""
        teams = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:vs?\.?|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', content)
        scores = re.findall(r'\b(\d+)\s*[-:]\s*(\d+)\b', content)
        venues = re.findall(r'(?:at|@)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Stadium|Arena|Field|Center)?)', content)
        
        return {
            'matchups': [{'team1': match[0], 'team2': match[1]} for match in teams],
            'scores': [{'score1': score[0], 'score2': score[1]} for score in scores],
            'venues': list(set(venues)),
            'content_type': 'sports_schedule'
        }
    
    def _extract_command_data(self, content: str) -> Dict[str, Any]:
        """Extract command/reference data."""
        shell_commands = re.findall(r'\$\s+([^\n]+)', content)
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', content, re.DOTALL)
        options = re.findall(r'--?([a-zA-Z-]+)', content)
        
        return {
            'shell_commands': shell_commands,
            'code_examples': code_blocks,
            'command_options': list(set(options)),
            'content_type': 'reference_guide'
        }
    
    def _extract_contact_data(self, content: str) -> Dict[str, Any]:
        """Extract contact information."""
        names = re.findall(r'(?:Name|Contact):\s*([^\n]+)', content, re.IGNORECASE)
        companies = re.findall(r'(?:Company|Organization):\s*([^\n]+)', content, re.IGNORECASE)
        addresses = re.findall(r'(?:Address):\s*([^\n]+(?:\n[^\n]+)*)', content, re.IGNORECASE)
        
        return {
            'contact_names': names,
            'companies': companies,
            'addresses': addresses,
            'content_type': 'contact_info'
        }
    
    def _extract_product_data(self, content: str) -> Dict[str, Any]:
        """Extract product information."""
        products = re.findall(r'(?:Product|Item):\s*([^\n]+)', content, re.IGNORECASE)
        skus = re.findall(r'(?:SKU|Product Code):\s*([^\n]+)', content, re.IGNORECASE)
        prices = re.findall(r'\$([0-9,]+\.?[0-9]*)', content)
        categories = re.findall(r'(?:Category|Type):\s*([^\n]+)', content, re.IGNORECASE)
        
        return {
            'products': products,
            'skus': skus,
            'prices': prices,
            'categories': categories,
            'content_type': 'product_data'
        }
    
    def _extract_event_data(self, content: str) -> Dict[str, Any]:
        """Extract event information."""
        events = re.findall(r'(?:Event|Conference|Meeting):\s*([^\n]+)', content, re.IGNORECASE)
        locations = re.findall(r'(?:Location|Venue|Address):\s*([^\n]+)', content, re.IGNORECASE)
        times = re.findall(r'(?:Time|Schedule):\s*([^\n]+)', content, re.IGNORECASE)
        
        return {
            'events': events,
            'locations': locations,
            'times': times,
            'content_type': 'event_info'
        }
    
    def _clean_content(self, content: str) -> str:
        """Aggressively clean content and create a summary if too long."""
        
        # FIRST: Remove null bytes and other problematic characters
        content = content.replace('\x00', '')  # Remove null bytes
        content = content.replace('\ufffd', '')  # Remove replacement characters
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)  # Remove control characters
        
        # Split into lines and clean each one
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip all leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Remove multiple consecutive spaces within the line
            line = re.sub(r' +', ' ', line)
            
            # Only keep lines with substantial content (at least 5 characters and some letters)
            if len(line) >= 5 and re.search(r'[a-zA-Z]', line):
                cleaned_lines.append(line)
        
        # Join with single newlines
        content = '\n'.join(cleaned_lines)
        
        # If content is still too long (>5000 chars), create a summary
        if len(content) > 5000:
            content = self._create_content_summary(content)
        
        return content.strip()
    
    def _create_content_summary(self, content: str) -> str:
        """Create a summary of long content to avoid storing excessive text."""
        
        lines = content.split('\n')
        
        # Take first 20 lines
        first_lines = lines[:20]
        
        # Take last 10 lines  
        last_lines = lines[-10:] if len(lines) > 30 else []
        
        # Sample some middle lines
        middle_lines = []
        if len(lines) > 50:
            step = len(lines) // 10
            for i in range(30, len(lines) - 10, step):
                if len(middle_lines) < 10:
                    middle_lines.append(lines[i])
        
        # Combine sections
        summary_parts = []
        
        if first_lines:
            summary_parts.append("=== BEGINNING ===")
            summary_parts.extend(first_lines)
        
        if middle_lines:
            summary_parts.append("=== SAMPLE CONTENT ===")
            summary_parts.extend(middle_lines)
        
        if last_lines:
            summary_parts.append("=== END ===")
            summary_parts.extend(last_lines)
        
        summary_parts.append(f"=== SUMMARY: {len(lines)} total lines, showing key excerpts ===")
        
        return '\n'.join(summary_parts)

    def _calculate_intelligent_priority(self, content: str, signature: ContentSignature) -> float:
        """Calculate priority with intelligence about content type."""
        priority = 5.0  # Base priority
        
        # Urgent markers
        urgent_words = ['urgent', 'critical', 'emergency', 'immediate', 'asap']
        for word in urgent_words:
            if word in content.lower():
                priority += 2
        
        # Content-type specific priorities
        if 'kubernetes' in signature.content_markers:
            # Kubernetes incidents are higher priority
            if any(word in content.lower() for word in ['error', 'failed', 'down', 'issue']):
                priority += 3
        
        if 'schedule_format' in signature.structural_patterns:
            # Schedules with today/tomorrow dates are higher priority
            if any(word in content.lower() for word in ['today', 'tomorrow', 'this week']):
                priority += 2
        
        return min(priority, 10.0)
    
    def _generate_learning_feedback(self, signature: ContentSignature, 
                                  similar_content: List[Tuple[str, float]], 
                                  confidence: float) -> Dict[str, Any]:
        """Generate feedback about what the system learned."""
        
        feedback = {
            'confidence_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'similar_content_found': len(similar_content),
            'new_patterns_detected': list(signature.content_markers),
            'structural_insights': list(signature.structural_patterns),
            'domain_classification': list(signature.domain_keywords)[:5]  # Top 5 keywords
        }
        
        if similar_content:
            feedback['learning_note'] = f"Found {len(similar_content)} similar documents - improving classification accuracy"
        else:
            feedback['learning_note'] = "New content type detected - expanding knowledge base"
        
        return feedback
    
    def _map_table_to_content_type(self, table_name: str) -> str:
        """Map table name to content type."""
        mapping = {
            'contacts': 'contact_info',
            'business_cards': 'business_card', 
            'products': 'product_data',
            'events': 'event_info',
            'cooking_recipes': 'recipe',
            'transactions': 'financial_data',
            'employees': 'employee_data',
            'articles': 'article',
            'kubernetes_resources': 'technical_document',
            'technical_documents': 'technical_document',
            'technical_references': 'technical_reference',
            'sports_documents': 'sports_document',
            'schedule_documents': 'schedule',
            'general_documents': 'document'
        }
        return mapping.get(table_name, 'document')
    
    def _extract_domain_from_table(self, table_name: str) -> str:
        """Extract domain from table name."""
        if '_' in table_name:
            return table_name.split('_')[0]
        return table_name
    
    def _content_type_to_collection(self, content_type: ContentType) -> str:
        """Map content type to collection name."""
        mapping = {
            ContentType.CONTACT_INFO: 'contacts',
            ContentType.BUSINESS_CARD: 'business_cards',
            ContentType.PRODUCT_DATA: 'products',
            ContentType.EVENT_INFO: 'events',
            ContentType.RECIPE: 'cooking_recipes',
            ContentType.FINANCIAL_DATA: 'transactions',
            ContentType.EMPLOYEE_DATA: 'employees',
            ContentType.ARTICLE: 'articles',
            ContentType.LOG_ENTRIES: 'log_entries',
            ContentType.INVOICE: 'invoices',
            ContentType.EMAIL_THREAD: 'email_threads',
            ContentType.MEETING_NOTES: 'meeting_notes',
            ContentType.COVER_LETTER: 'cover_letters',
            ContentType.RESUME_CV: 'resumes',
            ContentType.PERSONAL_STATEMENT: 'personal_statements',
            ContentType.UNKNOWN: 'general_documents'
        }
        return mapping.get(content_type, 'general_documents')
    
    def _fallback_analysis(self, content: str, filename: str) -> IntelligentResult:
        """Fallback when models aren't available."""
        
        signature = ContentSignature(
            domain_keywords=set(),
            structural_patterns=set(),
            entity_types=set(),
            content_markers=set(),
            semantic_hash=hashlib.md5(content[:100].encode()).hexdigest()
        )
        
        schema_evolution = TableEvolution(
            table_name='general_documents',
            current_schema={},
            suggested_additions={},
            pattern_frequency={},
            confidence_score=0.3,
            sample_contents=[content[:200]]
        )
        
        return IntelligentResult(
            table_name='general_documents',
            confidence=0.3,
            reasoning="Fallback analysis - NLP models not available",
            schema_evolution=schema_evolution,
            extracted_data={
                'content': content,
                'source_file': filename,
                'extracted_at': datetime.utcnow().isoformat(),
                'word_count': len(content.split())
            },
            similar_content_ids=[],
            learning_feedback={'note': 'Limited analysis without NLP models'}
        )