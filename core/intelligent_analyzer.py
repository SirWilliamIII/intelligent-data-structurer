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
    
    def analyze_content_signature(self, content: str, entities: List[Dict]) -> ContentSignature:
        """Create a semantic signature for content similarity matching."""
        
        # Extract domain keywords (meaningful nouns, verbs, technical terms)
        domain_keywords = set()
        content_lower = content.lower()
        
        # Technical domain indicators
        tech_patterns = ['command', 'server', 'api', 'database', 'config', 'install', 'error']
        sports_patterns = ['team', 'game', 'match', 'player', 'score', 'season', 'league']
        cooking_patterns = ['recipe', 'ingredient', 'cook', 'bake', 'serve', 'tablespoon', 'cup']
        
        for pattern in tech_patterns + sports_patterns + cooking_patterns:
            if pattern in content_lower:
                domain_keywords.add(pattern)
        
        # Add entity-based keywords
        for entity in entities:
            if entity['label'] in ['ORG', 'PRODUCT', 'TECHNOLOGY']:
                domain_keywords.add(entity['text'].lower())
        
        # Structural patterns
        structural_patterns = set()
        
        # List structures
        if re.search(r'\n\s*[-*•]\s+', content) or re.search(r'\n\s*\d+\.\s+', content):
            structural_patterns.add('list_format')
        
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
        
        # Entity types
        entity_types = set(entity['label'] for entity in entities)
        
        # Content markers (specific identifiers)
        content_markers = set()
        
        # Technology markers
        if 'kubernetes' in content_lower or 'kubectl' in content_lower:
            content_markers.add('kubernetes')
        if 'docker' in content_lower:
            content_markers.add('docker')
        if 'git' in content_lower:
            content_markers.add('git')
        
        # Sports markers
        if any(sport in content_lower for sport in ['football', 'soccer', 'basketball', 'baseball']):
            content_markers.add('sports_schedule')
        
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
        signature = self.learning_system.analyze_content_signature(content, entities)
        
        # Find similar content
        similar_content = self.learning_system.find_similar_content(signature)
        
        # Determine table name intelligently
        table_name, confidence, reasoning = self._determine_intelligent_table(
            signature, similar_content, content, filename
        )
        
        # Extract comprehensive data
        extracted_data = self._extract_intelligent_data(content, entities, signature, filename)
        
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
                                   content: str, filename: str) -> Tuple[str, float, str]:
        """Intelligently determine table name using learning and similarity."""
        
        reasoning_parts = []
        
        # If we have very similar content, use the same table
        if similar_content and similar_content[0][1] > 0.8:
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
        
        # Content marker-based naming (most specific)
        if 'kubernetes' in signature.content_markers:
            reasoning_parts.append("Kubernetes-specific content detected")
            return 'kubernetes_resources', 0.9, "; ".join(reasoning_parts)
        
        if 'docker' in signature.content_markers:
            reasoning_parts.append("Docker-specific content detected")
            return 'docker_references', 0.9, "; ".join(reasoning_parts)
        
        if 'sports_schedule' in signature.content_markers:
            reasoning_parts.append("Sports schedule format detected")
            return 'sports_schedules', 0.85, "; ".join(reasoning_parts)
        
        # Structural pattern-based naming
        if 'recipe_format' in signature.structural_patterns:
            reasoning_parts.append("Recipe structure detected")
            return 'cooking_recipes', 0.8, "; ".join(reasoning_parts)
        
        if 'command_format' in signature.structural_patterns:
            reasoning_parts.append("Command/technical reference format")
            return 'technical_references', 0.75, "; ".join(reasoning_parts)
        
        if 'schedule_format' in signature.structural_patterns:
            reasoning_parts.append("Schedule/calendar format detected")
            return 'event_schedules', 0.7, "; ".join(reasoning_parts)
        
        # Domain-based naming (more general)
        domain_keywords = signature.domain_keywords
        
        if any(kw in domain_keywords for kw in ['command', 'server', 'api', 'database']):
            reasoning_parts.append("Technical domain keywords detected")
            return 'technical_documents', 0.6, "; ".join(reasoning_parts)
        
        if any(kw in domain_keywords for kw in ['team', 'game', 'match', 'player']):
            reasoning_parts.append("Sports domain keywords detected")
            return 'sports_documents', 0.6, "; ".join(reasoning_parts)
        
        if any(kw in domain_keywords for kw in ['recipe', 'ingredient', 'cook']):
            reasoning_parts.append("Cooking domain keywords detected")
            return 'cooking_documents', 0.6, "; ".join(reasoning_parts)
        
        # Filename-based fallback
        filename_lower = filename.lower()
        if 'cheat' in filename_lower or 'reference' in filename_lower:
            reasoning_parts.append("Reference document based on filename")
            return 'reference_documents', 0.5, "; ".join(reasoning_parts)
        
        if 'schedule' in filename_lower or 'calendar' in filename_lower:
            reasoning_parts.append("Schedule document based on filename")
            return 'schedule_documents', 0.5, "; ".join(reasoning_parts)
        
        # Default
        reasoning_parts.append("General document - no specific patterns detected")
        return 'general_documents', 0.3, "; ".join(reasoning_parts)
    
    def _extract_intelligent_data(self, content: str, entities: List[Dict], 
                                signature: ContentSignature, filename: str) -> Dict[str, Any]:
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