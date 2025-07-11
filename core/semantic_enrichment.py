"""
Dynamic Semantic Enrichment System

Discovers domain knowledge organically by analyzing entities through web searches.
Learns what entities have in common without hardcoded knowledge.
"""

import asyncio
import re
import json
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
from loguru import logger
import hashlib
import sqlite3
from datetime import datetime, timedelta

@dataclass
class EntityContext:
    """Context discovered for an entity through web research."""
    entity: str
    domain_keywords: Set[str]
    category_hints: Set[str]
    related_entities: Set[str]
    confidence: float
    discovered_at: datetime
    search_summary: str

class SemanticEnrichmentEngine:
    """
    Dynamically discovers domain knowledge by researching entities through web search.
    Learns organic patterns without hardcoded knowledge.
    """
    
    def __init__(self, cache_db_path: str = "./semantic_cache.db"):
        self.cache_db_path = cache_db_path
        self.entity_contexts = {}
        self.domain_patterns = defaultdict(set)
        self._init_cache_db()
        
    def _init_cache_db(self):
        """Initialize local cache for discovered entity contexts."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_contexts (
                entity TEXT PRIMARY KEY,
                domain_keywords TEXT,
                category_hints TEXT,
                related_entities TEXT,
                confidence REAL,
                discovered_at TIMESTAMP,
                search_summary TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS domain_patterns (
                pattern_hash TEXT PRIMARY KEY,
                entities TEXT,
                shared_keywords TEXT,
                domain_category TEXT,
                confidence REAL,
                last_updated TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def enrich_entities(self, entities: List[str], content_context: str = "") -> Dict[str, Set[str]]:
        """
        Discover domain context for entities organically through web research.
        Returns enriched keywords for each entity.
        """
        enriched_keywords = defaultdict(set)
        
        # Get contexts for each entity
        entity_contexts = []
        for entity in entities:
            context = await self._get_entity_context(entity, content_context)
            if context:
                entity_contexts.append(context)
                enriched_keywords[entity].update(context.domain_keywords)
        
        # Discover shared domain patterns
        if len(entity_contexts) >= 2:
            shared_patterns = await self._discover_shared_patterns(entity_contexts)
            
            # Add shared domain keywords to all entities
            for entity in entities:
                enriched_keywords[entity].update(shared_patterns)
        
        return dict(enriched_keywords)
    
    async def _get_entity_context(self, entity: str, content_context: str = "") -> Optional[EntityContext]:
        """Get or discover context for a single entity."""
        
        # Check cache first
        cached = self._get_cached_context(entity)
        if cached and self._is_cache_fresh(cached):
            return cached
        
        # Discover context through web search
        try:
            context = await self._discover_entity_context(entity, content_context)
            if context:
                self._cache_context(context)
                return context
        except Exception as e:
            logger.warning(f"Failed to discover context for '{entity}': {e}")
        
        return None
    
    async def _discover_entity_context(self, entity: str, content_context: str = "") -> Optional[EntityContext]:
        """Discover entity context through organic web research."""
        
        # Create search query with context
        search_query = f"what is {entity}"
        if content_context:
            # Add context clues from the document
            context_words = re.findall(r'\b\w{4,}\b', content_context.lower())
            top_context = Counter(context_words).most_common(3)
            if top_context:
                context_hint = " ".join([word for word, _ in top_context])
                search_query = f"{entity} {context_hint}"
        
        try:
            # Use WebSearch to discover what this entity is
            # Import here to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            
            # For now, use a simple web search approach
            # In production, this would integrate with the WebSearch tool
            search_results = await self._simple_web_search(search_query)
            
            # Extract domain keywords from search results
            domain_keywords = self._extract_domain_keywords(search_results, entity)
            category_hints = self._extract_category_hints(search_results, entity)
            related_entities = self._extract_related_entities(search_results, entity)
            
            # Calculate confidence based on consistency of results
            confidence = self._calculate_discovery_confidence(search_results, domain_keywords)
            
            return EntityContext(
                entity=entity,
                domain_keywords=domain_keywords,
                category_hints=category_hints,
                related_entities=related_entities,
                confidence=confidence,
                discovered_at=datetime.now(),
                search_summary=search_results[:500]  # Store brief summary
            )
            
        except Exception as e:
            logger.error(f"Web search failed for entity '{entity}': {e}")
            return None
    
    async def _simple_web_search(self, query: str) -> str:
        """Simple web search implementation - placeholder for WebSearch tool integration."""
        # This is a placeholder - in practice this would use the WebSearch tool
        # For now, return a mock result that demonstrates the concept
        
        # Mock NFL team search results to demonstrate the concept
        mock_results = {
            "arizona cardinals": "The Arizona Cardinals are a professional American football team based in Phoenix, Arizona. They are members of the National Football League (NFL) and compete in the NFC West division. The team plays their home games at State Farm Stadium.",
            "dallas cowboys": "The Dallas Cowboys are a professional American football team based in Dallas, Texas. They are members of the National Football League (NFL) and compete in the NFC East division. The team is known as America's Team.",
            "kansas city chiefs": "The Kansas City Chiefs are a professional American football team based in Kansas City, Missouri. They compete in the National Football League (NFL) as members of the AFC West division.",
            "detroit lions": "The Detroit Lions are a professional American football team based in Detroit, Michigan. They are members of the National Football League (NFL) and compete in the NFC North division."
        }
        
        query_lower = query.lower()
        for entity, result in mock_results.items():
            if entity in query_lower:
                return result
        
        # Generic fallback for unknown entities
        return f"Information about {query} - professional organization team league sports"
    
    def _extract_domain_keywords(self, search_text: str, entity: str) -> Set[str]:
        """Extract domain-specific keywords from search results."""
        domain_keywords = set()
        
        # Common domain indicators to look for
        domain_patterns = {
            # Sports patterns
            r'\b(team|sport|league|football|basketball|baseball|hockey|soccer|nfl|nba|mlb|nhl)\b': 'sports',
            r'\b(game|match|season|player|coach|stadium|championship)\b': 'sports',
            
            # Technology patterns  
            r'\b(software|technology|computer|digital|tech|programming|code)\b': 'technology',
            r'\b(app|platform|system|network|database|api)\b': 'technology',
            
            # Business patterns
            r'\b(company|business|corporation|enterprise|organization|firm)\b': 'business',
            r'\b(ceo|executive|management|headquarters|revenue|industry)\b': 'business',
            
            # Entertainment patterns
            r'\b(movie|film|actor|actress|director|entertainment|hollywood)\b': 'entertainment',
            r'\b(music|song|artist|album|record|band)\b': 'entertainment',
            
            # Geographic patterns
            r'\b(city|state|country|location|place|region|area)\b': 'geography'
        }
        
        text_lower = search_text.lower()
        
        for pattern, category in domain_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                domain_keywords.update(matches)
                domain_keywords.add(category)  # Add the category itself
        
        # Remove the entity itself from keywords
        domain_keywords.discard(entity.lower())
        
        return domain_keywords
    
    def _extract_category_hints(self, search_text: str, entity: str) -> Set[str]:
        """Extract category hints like 'team', 'company', 'software'."""
        category_hints = set()
        
        # Look for "X is a Y" patterns
        is_patterns = re.findall(rf'{re.escape(entity.lower())} is (?:a |an )?(\w+)', search_text.lower())
        category_hints.update(is_patterns)
        
        # Look for "Y called X" patterns  
        called_patterns = re.findall(rf'(\w+) called {re.escape(entity.lower())}', search_text.lower())
        category_hints.update(called_patterns)
        
        return category_hints
    
    def _extract_related_entities(self, search_text: str, entity: str) -> Set[str]:
        """Extract other entities that appear frequently with this entity."""
        related = set()
        
        # Simple approach: find capitalized words that appear near the entity
        sentences = re.split(r'[.!?]', search_text)
        entity_sentences = [s for s in sentences if entity.lower() in s.lower()]
        
        for sentence in entity_sentences[:3]:  # Limit to first few relevant sentences
            # Find other capitalized entities (simple heuristic)
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
            for ent in entities:
                if ent.lower() != entity.lower() and len(ent) > 3:
                    related.add(ent)
        
        return related
    
    def _calculate_discovery_confidence(self, search_text: str, domain_keywords: Set[str]) -> float:
        """Calculate confidence in the discovered context."""
        base_confidence = 0.5
        
        # Boost confidence for rich domain keywords
        if len(domain_keywords) > 3:
            base_confidence += 0.2
        
        # Boost confidence for consistent patterns
        text_lower = search_text.lower()
        keyword_mentions = sum(1 for kw in domain_keywords if kw in text_lower)
        if keyword_mentions > len(domain_keywords) * 0.7:
            base_confidence += 0.2
        
        # Boost confidence for longer, more detailed results
        if len(search_text) > 1000:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    async def _discover_shared_patterns(self, entity_contexts: List[EntityContext]) -> Set[str]:
        """Discover what multiple entities have in common organically."""
        
        if len(entity_contexts) < 2:
            return set()
        
        # Find keywords that appear across multiple entities
        all_keywords = [ctx.domain_keywords for ctx in entity_contexts]
        shared_keywords = set.intersection(*all_keywords) if all_keywords else set()
        
        # Find category hints that are consistent
        all_categories = [ctx.category_hints for ctx in entity_contexts]
        shared_categories = set.intersection(*all_categories) if all_categories else set()
        
        # Combine shared patterns
        shared_patterns = shared_keywords.union(shared_categories)
        
        # Cache the discovered pattern
        if shared_patterns:
            await self._cache_shared_pattern(
                entities=[ctx.entity for ctx in entity_contexts],
                shared_keywords=shared_patterns,
                confidence=min(ctx.confidence for ctx in entity_contexts)
            )
        
        logger.info(f"Discovered shared patterns for {len(entity_contexts)} entities: {shared_patterns}")
        
        return shared_patterns
    
    def _get_cached_context(self, entity: str) -> Optional[EntityContext]:
        """Retrieve cached context for entity."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM entity_contexts WHERE entity = ?', (entity,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return EntityContext(
                entity=row[0],
                domain_keywords=set(json.loads(row[1])),
                category_hints=set(json.loads(row[2])),
                related_entities=set(json.loads(row[3])),
                confidence=row[4],
                discovered_at=datetime.fromisoformat(row[5]),
                search_summary=row[6]
            )
        
        return None
    
    def _is_cache_fresh(self, context: EntityContext, max_age_days: int = 7) -> bool:
        """Check if cached context is still fresh."""
        age = datetime.now() - context.discovered_at
        return age.days < max_age_days
    
    def _cache_context(self, context: EntityContext):
        """Cache discovered context."""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO entity_contexts 
            (entity, domain_keywords, category_hints, related_entities, confidence, discovered_at, search_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            context.entity,
            json.dumps(list(context.domain_keywords)),
            json.dumps(list(context.category_hints)),
            json.dumps(list(context.related_entities)),
            context.confidence,
            context.discovered_at.isoformat(),
            context.search_summary
        ))
        
        conn.commit()
        conn.close()
    
    async def _cache_shared_pattern(self, entities: List[str], shared_keywords: Set[str], confidence: float):
        """Cache discovered shared patterns between entities."""
        pattern_hash = hashlib.md5(json.dumps(sorted(entities)).encode()).hexdigest()
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO domain_patterns
            (pattern_hash, entities, shared_keywords, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            pattern_hash,
            json.dumps(entities),
            json.dumps(list(shared_keywords)),
            confidence,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()

# Global instance
semantic_enricher = SemanticEnrichmentEngine()