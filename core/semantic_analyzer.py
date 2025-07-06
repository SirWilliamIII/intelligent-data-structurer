"""
Semantic analyzer for domain-agnostic dynamic table creation.
Handles any type of content and creates appropriate tables.
"""

import re
import json
import spacy
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from loguru import logger
from collections import Counter

@dataclass
class SemanticResult:
    """Semantic analysis result for any content type."""
    table_name: str
    domain: str              # technical, sports, cooking, business, etc.
    content_type: str        # cheat_sheet, schedule, recipe, meeting_notes, etc.
    priority: float          # 0-10 based on urgency/importance indicators
    confidence: float        # How confident we are in the classification
    reasoning: str          # Why this classification was chosen
    schema_hints: Dict[str, str]  # Suggested additional columns for this content type
    extracted_data: Dict[str, Any]  # The actual structured data

class SemanticAnalyzer:
    """Domain-agnostic content analyzer that creates tables based on semantic understanding."""
    
    def __init__(self):
        self.nlp = None
        self._load_models()
        
        # Domain patterns - these help identify what domain the content belongs to
        self.domain_patterns = {
            'technical': [
                'kubernetes', 'docker', 'api', 'server', 'database', 'code', 'command',
                'install', 'configure', 'debug', 'error', 'log', 'system', 'network',
                'script', 'terminal', 'shell', 'git', 'deployment', 'container'
            ],
            'sports': [
                'team', 'game', 'match', 'season', 'league', 'tournament', 'score',
                'player', 'coach', 'stadium', 'schedule', 'fixture', 'championship'
            ],
            'cooking': [
                'recipe', 'ingredients', 'cook', 'bake', 'serve', 'tablespoon', 'cup',
                'minutes', 'temperature', 'oven', 'heat', 'mix', 'add', 'prepare'
            ],
            'business': [
                'meeting', 'agenda', 'project', 'deadline', 'budget', 'revenue',
                'employee', 'manager', 'department', 'policy', 'procedure', 'report'
            ],
            'education': [
                'course', 'lesson', 'student', 'teacher', 'assignment', 'exam',
                'grade', 'syllabus', 'homework', 'study', 'learn', 'tutorial'
            ],
            'healthcare': [
                'patient', 'doctor', 'treatment', 'diagnosis', 'medication', 'symptom',
                'medical', 'health', 'clinic', 'hospital', 'appointment', 'therapy'
            ],
            'travel': [
                'flight', 'hotel', 'destination', 'itinerary', 'booking', 'airport',
                'vacation', 'trip', 'travel', 'accommodation', 'tourist', 'visa'
            ],
            'entertainment': [
                'movie', 'music', 'concert', 'show', 'artist', 'album', 'song',
                'theater', 'performance', 'festival', 'event', 'ticket'
            ]
        }
        
        # Content type patterns - these help identify the structure/purpose of content
        self.content_type_patterns = {
            'cheat_sheet': [
                'cheat sheet', 'quick reference', 'commands', 'shortcuts', 'tips',
                'guide', 'reference', 'how to', 'examples'
            ],
            'schedule': [
                'schedule', 'calendar', 'timetable', 'agenda', 'timeline', 'fixture',
                'appointment', 'meeting', 'event', 'date', 'time'
            ],
            'tutorial': [
                'tutorial', 'step by step', 'guide', 'how to', 'instructions',
                'walkthrough', 'lesson', 'learning'
            ],
            'list': [
                'checklist', 'todo', 'list', 'items', 'inventory', 'catalog'
            ],
            'documentation': [
                'documentation', 'manual', 'specification', 'readme', 'docs',
                'overview', 'introduction', 'explanation'
            ],
            'notes': [
                'notes', 'memo', 'summary', 'minutes', 'record', 'log', 'journal'
            ],
            'policy': [
                'policy', 'procedure', 'rules', 'guidelines', 'standards',
                'requirements', 'regulations', 'compliance'
            ]
        }
        
    def _load_models(self):
        """Load NLP models."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Semantic NLP models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            self.nlp = None
    
    async def analyze_content(self, content: str, filename: str) -> SemanticResult:
        """Analyze content semantically to determine table and schema."""
        
        if not self.nlp:
            return self._fallback_analysis(content, filename)
        
        # Process with spaCy for entities and linguistic features
        doc = self.nlp(content)
        entities = self._extract_entities(doc)
        
        # Determine domain (technical, sports, cooking, etc.)
        domain, domain_confidence = self._identify_domain(content)
        
        # Determine content type (cheat_sheet, schedule, tutorial, etc.)
        content_type, type_confidence = self._identify_content_type(content, filename)
        
        # Calculate priority based on urgency/importance indicators
        priority = self._calculate_priority(content)
        
        # Generate table name
        table_name = self._generate_table_name(domain, content_type, content)
        
        # Suggest schema based on content analysis
        schema_hints = self._suggest_schema(domain, content_type, entities, content)
        
        # Extract structured data
        extracted_data = self._extract_structured_data(content, entities, domain, content_type, filename)
        
        # Overall confidence and reasoning
        confidence = min(domain_confidence, type_confidence)
        reasoning = f"Domain: {domain} ({domain_confidence:.1f}), Type: {content_type} ({type_confidence:.1f})"
        
        return SemanticResult(
            table_name=table_name,
            domain=domain,
            content_type=content_type,
            priority=priority,
            confidence=confidence,
            reasoning=reasoning,
            schema_hints=schema_hints,
            extracted_data=extracted_data
        )
    
    def _extract_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract named entities with context."""
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) or ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    
    def _identify_domain(self, content: str) -> Tuple[str, float]:
        """Identify the semantic domain of the content."""
        content_lower = content.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_patterns.items():
            score = 0
            for keyword in keywords:
                # Count occurrences, give higher weight to exact matches
                if keyword in content_lower:
                    score += content_lower.count(keyword)
                    # Bonus for keyword appearing in title/first paragraph
                    if keyword in content_lower[:200]:
                        score += 0.5
            
            domain_scores[domain] = score
        
        # Find the domain with highest score
        if not domain_scores or max(domain_scores.values()) == 0:
            return 'general', 0.3
        
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Normalize confidence (0-1)
        confidence = min(max_score / 10, 1.0)
        
        return best_domain, confidence
    
    def _identify_content_type(self, content: str, filename: str) -> Tuple[str, float]:
        """Identify the content type/structure."""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        type_scores = {}
        
        # Check filename for hints
        for content_type, keywords in self.content_type_patterns.items():
            score = 0
            
            # Check content
            for keyword in keywords:
                if keyword in content_lower:
                    score += content_lower.count(keyword)
            
            # Check filename (higher weight)
            for keyword in keywords:
                if keyword in filename_lower:
                    score += 2
            
            # Structural analysis
            if content_type == 'schedule' and self._has_schedule_structure(content):
                score += 3
            elif content_type == 'list' and self._has_list_structure(content):
                score += 3
            elif content_type == 'cheat_sheet' and self._has_reference_structure(content):
                score += 3
            
            type_scores[content_type] = score
        
        if not type_scores or max(type_scores.values()) == 0:
            return 'document', 0.3
        
        best_type = max(type_scores, key=type_scores.get)
        max_score = type_scores[best_type]
        
        confidence = min(max_score / 8, 1.0)
        
        return best_type, confidence
    
    def _has_schedule_structure(self, content: str) -> bool:
        """Check if content has schedule-like structure."""
        # Look for date/time patterns
        date_patterns = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content))
        time_patterns = len(re.findall(r'\b\d{1,2}:\d{2}\b', content))
        
        return date_patterns >= 2 or time_patterns >= 2
    
    def _has_list_structure(self, content: str) -> bool:
        """Check if content has list-like structure."""
        # Count bullet points, numbers, dashes
        bullets = content.count('•') + content.count('-') + content.count('*')
        numbered = len(re.findall(r'\n\s*\d+\.', content))
        
        return bullets >= 3 or numbered >= 3
    
    def _has_reference_structure(self, content: str) -> bool:
        """Check if content has reference/cheat sheet structure."""
        # Look for command patterns, code blocks, examples
        code_blocks = content.count('```') + content.count('`')
        command_patterns = len(re.findall(r'\$\s+\w+', content))
        
        return code_blocks >= 2 or command_patterns >= 3
    
    def _calculate_priority(self, content: str) -> float:
        """Calculate priority score (0-10)."""
        priority = 0
        content_lower = content.lower()
        
        # Urgency indicators
        urgent_words = ['urgent', 'immediate', 'asap', 'critical', 'emergency']
        for word in urgent_words:
            if word in content_lower:
                priority += 2
        
        # Important indicators
        important_words = ['important', 'priority', 'deadline', 'required']
        for word in important_words:
            if word in content_lower:
                priority += 1
        
        # Time sensitivity
        if re.search(r'\b(today|tomorrow|this week)\b', content_lower):
            priority += 1
        
        return min(priority, 10)
    
    def _generate_table_name(self, domain: str, content_type: str, content: str) -> str:
        """Generate appropriate table name."""
        # For specific content types, create more descriptive names
        if domain == 'technical' and content_type == 'cheat_sheet':
            if 'kubernetes' in content.lower():
                return 'kubernetes_references'
            elif 'docker' in content.lower():
                return 'docker_references'
            else:
                return 'technical_references'
        
        elif domain == 'sports' and content_type == 'schedule':
            return 'sports_schedules'
        
        elif domain == 'cooking' and content_type in ['document', 'tutorial']:
            return 'recipes'
        
        elif content_type == 'schedule':
            return f'{domain}_schedules'
        
        elif content_type == 'cheat_sheet':
            return f'{domain}_references'
        
        elif content_type == 'notes':
            return f'{domain}_notes'
        
        elif content_type == 'policy':
            return f'{domain}_policies'
        
        else:
            # Generic naming
            return f'{domain}_{content_type}s'
    
    def _suggest_schema(self, domain: str, content_type: str, entities: List[Dict], content: str) -> Dict[str, str]:
        """Suggest additional schema columns based on content analysis."""
        schema = {}
        
        # Base schema that all tables get
        base_schema = {
            'priority': 'FLOAT',
            'domain': 'VARCHAR(50)',
            'content_type': 'VARCHAR(50)'
        }
        schema.update(base_schema)
        
        # Domain-specific columns
        if domain == 'technical':
            schema.update({
                'technology': 'VARCHAR(100)',
                'commands': 'JSONB',
                'difficulty_level': 'VARCHAR(20)'
            })
        
        elif domain == 'sports':
            schema.update({
                'sport_type': 'VARCHAR(50)',
                'team_names': 'JSONB',
                'venue': 'VARCHAR(200)',
                'season': 'VARCHAR(20)'
            })
        
        elif domain == 'cooking':
            schema.update({
                'prep_time': 'VARCHAR(20)',
                'cook_time': 'VARCHAR(20)',
                'servings': 'INTEGER',
                'ingredients': 'JSONB',
                'instructions': 'JSONB'
            })
        
        # Content type specific columns
        if content_type == 'schedule':
            schema.update({
                'event_date': 'DATE',
                'event_time': 'TIME',
                'location': 'VARCHAR(200)',
                'participants': 'JSONB'
            })
        
        elif content_type == 'cheat_sheet':
            schema.update({
                'commands': 'JSONB',
                'examples': 'JSONB',
                'tips': 'JSONB'
            })
        
        # Add columns based on detected entities
        person_entities = [e for e in entities if e['label'] == 'PERSON']
        if person_entities:
            schema['people_involved'] = 'JSONB'
        
        date_entities = [e for e in entities if e['label'] == 'DATE']
        if date_entities:
            schema['important_dates'] = 'JSONB'
        
        money_entities = [e for e in entities if e['label'] == 'MONEY']
        if money_entities:
            schema['financial_amounts'] = 'JSONB'
        
        return schema
    
    def _extract_structured_data(self, content: str, entities: List[Dict], 
                                domain: str, content_type: str, filename: str) -> Dict[str, Any]:
        """Extract structured data with domain and content-type awareness."""
        
        # Base extraction
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        phones = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', content)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b', content)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        
        data = {
            'content': content,
            'entities': entities,
            'emails': emails,
            'phones': phones,
            'dates': dates,
            'urls': urls,
            'word_count': len(content.split()),
            'char_count': len(content),
            'source_file': filename,
            'extracted_at': datetime.utcnow().isoformat(),
            'domain': domain,
            'content_type': content_type,
            'priority': self._calculate_priority(content)
        }
        
        # Domain-specific extraction
        if domain == 'technical':
            data.update(self._extract_technical_data(content))
        elif domain == 'sports':
            data.update(self._extract_sports_data(content))
        elif domain == 'cooking':
            data.update(self._extract_cooking_data(content))
        
        # Content-type specific extraction
        if content_type == 'cheat_sheet':
            data.update(self._extract_reference_data(content))
        elif content_type == 'schedule':
            data.update(self._extract_schedule_data(content))
        
        return data
    
    def _extract_technical_data(self, content: str) -> Dict[str, Any]:
        """Extract technical-specific data."""
        commands = re.findall(r'\$\s+([^\n]+)', content)
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', content, re.DOTALL)
        
        return {
            'commands': commands,
            'code_blocks': code_blocks,
            'technology': self._identify_technology(content)
        }
    
    def _extract_sports_data(self, content: str) -> Dict[str, Any]:
        """Extract sports-specific data."""
        teams = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', content)  # Team names pattern
        scores = re.findall(r'\b\d+[-:]\d+\b', content)
        
        return {
            'teams': list(set(teams)),
            'scores': scores
        }
    
    def _extract_cooking_data(self, content: str) -> Dict[str, Any]:
        """Extract cooking-specific data."""
        # Look for ingredient patterns
        ingredients = re.findall(r'\d+(?:\.\d+)?\s*(?:cups?|tablespoons?|teaspoons?|lbs?|oz|grams?)\s+([^\n,]+)', content, re.IGNORECASE)
        times = re.findall(r'\d+\s*(?:minutes?|hours?|mins?|hrs?)', content, re.IGNORECASE)
        
        return {
            'ingredients': ingredients,
            'cooking_times': times
        }
    
    def _extract_reference_data(self, content: str) -> Dict[str, Any]:
        """Extract reference/cheat sheet specific data."""
        examples = re.findall(r'example[s]?:?\s*([^\n]+)', content, re.IGNORECASE)
        tips = re.findall(r'tip[s]?:?\s*([^\n]+)', content, re.IGNORECASE)
        
        return {
            'examples': examples,
            'tips': tips
        }
    
    def _extract_schedule_data(self, content: str) -> Dict[str, Any]:
        """Extract schedule-specific data."""
        events = []
        lines = content.split('\n')
        
        for line in lines:
            if re.search(r'\d{1,2}:\d{2}', line) or re.search(r'\d{1,2}[/-]\d{1,2}', line):
                events.append(line.strip())
        
        return {
            'events': events
        }
    
    def _identify_technology(self, content: str) -> str:
        """Identify specific technology from content."""
        tech_patterns = {
            'kubernetes': ['kubectl', 'k8s', 'kubernetes', 'pod', 'deployment'],
            'docker': ['docker', 'container', 'dockerfile', 'image'],
            'git': ['git', 'commit', 'branch', 'merge', 'repository'],
            'python': ['python', 'pip', 'import', 'def ', 'class '],
            'javascript': ['javascript', 'npm', 'node', 'function', 'const ']
        }
        
        content_lower = content.lower()
        for tech, keywords in tech_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                return tech
        
        return 'general'
    
    def _fallback_analysis(self, content: str, filename: str) -> SemanticResult:
        """Fallback when NLP isn't available."""
        return SemanticResult(
            table_name='general_documents',
            domain='general',
            content_type='document',
            priority=5.0,
            confidence=0.3,
            reasoning="Fallback analysis - NLP models not available",
            schema_hints={'priority': 'FLOAT', 'domain': 'VARCHAR(50)'},
            extracted_data={
                'content': content,
                'source_file': filename,
                'extracted_at': datetime.utcnow().isoformat(),
                'word_count': len(content.split()),
                'char_count': len(content)
            }
        )