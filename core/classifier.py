"""
Intelligent content classification system with confidence scoring.
"""

import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import asyncio

class ContentType(Enum):
    """Supported content types for automatic processing."""
    CONTACT_INFO = "contact_info"
    PRODUCT_DATA = "product_data"
    EVENT_INFO = "event_info"
    ARTICLE = "article"
    RECIPE = "recipe"
    FINANCIAL_DATA = "financial_data"
    LOG_ENTRIES = "log_entries"
    EMPLOYEE_DATA = "employee_data"
    BUSINESS_CARD = "business_card"
    INVOICE = "invoice"
    EMAIL_THREAD = "email_thread"
    MEETING_NOTES = "meeting_notes"
    COVER_LETTER = "cover_letter"
    RESUME_CV = "resume_cv"
    PERSONAL_STATEMENT = "personal_statement"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    """Result of content classification."""
    content_type: ContentType
    confidence: float
    reasoning: str
    suggested_action: str
    alternative_types: List[Tuple[ContentType, float]]

class IntelligentClassifier:
    """Advanced content classifier with multiple strategies."""
    
    def __init__(self):
        self.nlp = None
        self._load_models()
        
        # Enhanced pattern definitions
        self.patterns = {
            ContentType.CONTACT_INFO: {
                'keywords': ['email', 'phone', 'address', 'contact', 'mobile', 'tel', 'fax'],
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
                    r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',  # phone
                    r'\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd)\b',  # address
                    r'\b(mr|mrs|ms|dr|prof)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # formal names
                ],
                'weight': 3,
                'min_confidence': 0.3
            },
            
            ContentType.BUSINESS_CARD: {
                'keywords': ['ceo', 'manager', 'director', 'president', 'vp', 'company', 'corp', 'inc', 'llc'],
                'patterns': [
                    r'\b(ceo|cto|cfo|vp|president|director|manager)\b',
                    r'\b\w+\s+(corp|corporation|inc|llc|ltd|company)\b',
                    r'\bwww\.\w+\.(com|org|net)\b',
                ],
                'weight': 2,
                'min_confidence': 0.3
            },
            
            ContentType.PRODUCT_DATA: {
                'keywords': ['price', 'product', 'sku', 'inventory', 'item', 'cost', 'sale', 'discount'],
                'patterns': [
                    r'\$\d+\.?\d*|\d+\.\d{2}\s*(usd|dollars?)',  # price
                    r'\bsku\s*:?\s*\w+',  # SKU
                    r'\bmodel\s*:?\s*\w+',  # model number
                    r'\bin\s+stock\s*:?\s*\d+',  # inventory
                ],
                'weight': 2,
                'min_confidence': 0.3
            },
            
            ContentType.INVOICE: {
                'keywords': ['invoice', 'bill', 'total', 'due', 'payment', 'amount', 'tax', 'subtotal'],
                'patterns': [
                    r'\binvoice\s*#?\s*\d+',
                    r'\btotal\s*:?\s*\$\d+',
                    r'\bdue\s+date\s*:?',
                    r'\btax\s*:?\s*\$?\d+',
                ],
                'weight': 3,
                'min_confidence': 0.8
            },
            
            ContentType.EVENT_INFO: {
                'keywords': ['event', 'meeting', 'conference', 'workshop', 'seminar', 'venue', 'agenda'],
                'patterns': [
                    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',  # date
                    r'\b\d{1,2}:\d{2}\s*(am|pm)?\b',  # time
                    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',  # weekdays
                    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # months
                ],
                'weight': 2,
                'min_confidence': 0.3
            },
            
            ContentType.ARTICLE: {
                'keywords': ['article', 'author', 'published', 'title', 'abstract', 'conclusion'],
                'patterns': [
                    r'^#\s+.+$',  # markdown title
                    r'\bby\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # author
                    r'\bpublished\s*:',  # published field
                    r'\babstract\s*:',  # abstract
                ],
                'weight': 1,
                'min_confidence': 0.3
            },
            
            ContentType.RECIPE: {
                'keywords': ['recipe', 'ingredients', 'instructions', 'cook', 'bake', 'prep', 'serves'],
                'patterns': [
                    r'\bingredients?\s*:',
                    r'\binstructions?\s*:',
                    r'\bprep\s+time\s*:',
                    r'\bcook\s+time\s*:',
                    r'\bserves?\s*:?\s*\d+',
                    r'\b\d+\s+(cup|cups|tsp|tbsp|lb|oz|gram|ml)\b',  # measurements
                ],
                'weight': 2,
                'min_confidence': 0.3
            },
            
            ContentType.FINANCIAL_DATA: {
                'keywords': ['transaction', 'account', 'balance', 'deposit', 'withdrawal', 'transfer'],
                'patterns': [
                    r'\$\d+\.\d{2}',  # currency
                    r'\btransaction\s+id\s*:?\s*\w+',
                    r'\baccount\s+number\s*:?\s*\d+',
                    r'\bbalance\s*:?\s*\$',
                    r'\b(debit|credit|transfer)\b',
                ],
                'weight': 3,
                'min_confidence': 0.8
            },
            
            ContentType.LOG_ENTRIES: {
                'keywords': ['error', 'info', 'debug', 'warn', 'log', 'timestamp', 'exception'],
                'patterns': [
                    r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # timestamp
                    r'\[(ERROR|INFO|DEBUG|WARN|FATAL)\]',  # log level
                    r'^\d{4}-\d{2}-\d{2}',  # date at start
                    r'\bexception\b|\bstack\s+trace\b',  # errors
                ],
                'weight': 2,
                'min_confidence': 0.3
            },
            
            ContentType.EMPLOYEE_DATA: {
                'keywords': ['employee', 'department', 'salary', 'hire', 'position', 'manager', 'hr'],
                'patterns': [
                    r'\bemployee\s+id\s*:?\s*\d+',
                    r'\bdepartment\s*:',
                    r'\bsalary\s*:?\s*\$',
                    r'\bhire\s+date\s*:',
                    r'\bposition\s*:',
                ],
                'weight': 2,
                'min_confidence': 0.3
            },
            
            ContentType.EMAIL_THREAD: {
                'keywords': ['from', 'to', 'subject', 'reply', 'forward', 'sent'],
                'patterns': [
                    r'^from\s*:',
                    r'^to\s*:',
                    r'^subject\s*:',
                    r'^date\s*:',
                    r'wrote\s*:$',  # email reply indicator
                ],
                'weight': 2,
                'min_confidence': 0.8
            },
            
            ContentType.MEETING_NOTES: {
                'keywords': ['meeting', 'agenda', 'action', 'attendees', 'minutes', 'notes'],
                'patterns': [
                    r'\bmeeting\s+(notes|minutes)\b',
                    r'\battendees?\s*:',
                    r'\bagenda\s*:',
                    r'\baction\s+items?\s*:',
                    r'^\s*[-*]\s+',  # bullet points
                ],
                'weight': 2,
                'min_confidence': 0.3
            }
        }
    
    def _load_models(self):
        """Load NLP models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model successfully")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    async def classify_content(self, content: str, filename: str = "") -> ClassificationResult:
        """Classify content with confidence scoring."""
        
        # Multi-strategy classification
        rule_based_result = self._rule_based_classification(content, filename)
        
        if self.nlp:
            nlp_result = await self._nlp_based_classification(content)
            # Combine results
            final_result = self._combine_classifications(rule_based_result, nlp_result)
        else:
            final_result = rule_based_result
        
        logger.info(f"Classified content as {final_result.content_type.value} with confidence {final_result.confidence:.2f}")
        
        return final_result
    
    def _rule_based_classification(self, content: str, filename: str) -> ClassificationResult:
        """Rule-based classification using patterns and keywords."""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        scores = {}
        reasoning_parts = []
        
        for content_type, rules in self.patterns.items():
            score = 0
            matches = []
            
            # Check keywords
            keyword_matches = 0
            for keyword in rules['keywords']:
                if keyword in content_lower or keyword in filename_lower:
                    keyword_matches += 1
                    matches.append(f"keyword '{keyword}'")
            
            # Check patterns
            pattern_matches = 0
            for pattern in rules['patterns']:
                if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                    pattern_matches += 1
                    matches.append(f"pattern match")
            
            # Calculate weighted score
            total_score = (keyword_matches * 1) + (pattern_matches * 2)
            weighted_score = total_score * rules['weight']
            
            if matches:
                reasoning_parts.append(f"{content_type.value}: {', '.join(matches[:3])}")
            
            scores[content_type] = weighted_score
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        best_type = sorted_scores[0][0]
        best_score = sorted_scores[0][1]
        
        # Normalize confidence (0-1)
        max_possible_score = len(self.patterns[best_type]['keywords']) + (len(self.patterns[best_type]['patterns']) * 2)
        max_possible_score *= self.patterns[best_type]['weight']
        
        confidence = min(best_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        
        # Alternative types
        alternatives = [(t, s/max_possible_score) for t, s in sorted_scores[1:4] if s > 0]
        
        # Check minimum confidence threshold
        if confidence < self.patterns[best_type]['min_confidence']:
            best_type = ContentType.UNKNOWN
            confidence = 0.0
        
        reasoning = f"Rule-based analysis: {'; '.join(reasoning_parts[:3])}"
        
        return ClassificationResult(
            content_type=best_type,
            confidence=confidence,
            reasoning=reasoning,
            suggested_action=self._get_suggested_action(best_type, confidence),
            alternative_types=alternatives
        )
    
    async def _nlp_based_classification(self, content: str) -> ClassificationResult:
        """NLP-based classification using spaCy."""
        if not self.nlp:
            return ClassificationResult(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                reasoning="NLP model not available",
                suggested_action="Use rule-based classification only",
                alternative_types=[]
            )
        
        # Process text with spaCy
        doc = self.nlp(content[:5000])  # Limit text length for performance
        
        # Extract features
        entities = [(ent.label_, ent.text) for ent in doc.ents]
        pos_tags = [token.pos_ for token in doc]
        
        # Simple NLP-based scoring
        entity_scores = {
            ContentType.CONTACT_INFO: len([e for e in entities if e[0] in ['PERSON', 'ORG', 'GPE']]),
            ContentType.FINANCIAL_DATA: len([e for e in entities if e[0] in ['MONEY', 'PERCENT']]),
            ContentType.EVENT_INFO: len([e for e in entities if e[0] in ['DATE', 'TIME', 'EVENT']]),
            ContentType.ARTICLE: 1 if 'NOUN' in pos_tags and 'VERB' in pos_tags else 0,
        }
        
        best_type = max(entity_scores, key=entity_scores.get)
        confidence = min(entity_scores[best_type] / 10.0, 1.0)  # Normalize
        
        return ClassificationResult(
            content_type=best_type,
            confidence=confidence,
            reasoning=f"NLP analysis: found {len(entities)} entities",
            suggested_action=self._get_suggested_action(best_type, confidence),
            alternative_types=[]
        )
    
    def _combine_classifications(self, rule_result: ClassificationResult, nlp_result: ClassificationResult) -> ClassificationResult:
        """Combine rule-based and NLP results."""
        
        # Weight the results (rule-based gets more weight for now)
        rule_weight = 0.7
        nlp_weight = 0.3
        
        if rule_result.content_type == nlp_result.content_type:
            # Agreement - boost confidence
            final_confidence = min((rule_result.confidence * rule_weight + nlp_result.confidence * nlp_weight) * 1.2, 1.0)
            final_type = rule_result.content_type
            reasoning = f"Agreement: {rule_result.reasoning} + {nlp_result.reasoning}"
        else:
            # Disagreement - use higher confidence result
            if rule_result.confidence > nlp_result.confidence:
                final_type = rule_result.content_type
                final_confidence = rule_result.confidence * 0.8  # Reduce confidence due to disagreement
                reasoning = f"Rule-based winner: {rule_result.reasoning}"
            else:
                final_type = nlp_result.content_type
                final_confidence = nlp_result.confidence * 0.8
                reasoning = f"NLP-based winner: {nlp_result.reasoning}"
        
        return ClassificationResult(
            content_type=final_type,
            confidence=final_confidence,
            reasoning=reasoning,
            suggested_action=self._get_suggested_action(final_type, final_confidence),
            alternative_types=rule_result.alternative_types
        )
    
    def _get_suggested_action(self, content_type: ContentType, confidence: float) -> str:
        """Get suggested action based on classification and confidence."""
        
        if confidence >= 0.8:
            action_prefix = "Auto-process:"
        elif confidence >= 0.4:
            action_prefix = "Review and process:"
        else:
            action_prefix = "Manual classification needed:"
        
        actions = {
            ContentType.CONTACT_INFO: "Extract contact details and create contact records",
            ContentType.BUSINESS_CARD: "Parse business card info and create contact with company details",
            ContentType.PRODUCT_DATA: "Extract product information and update inventory",
            ContentType.INVOICE: "Parse invoice details and create financial records",
            ContentType.EVENT_INFO: "Create event records with proper scheduling",
            ContentType.ARTICLE: "Extract article metadata and create content records",
            ContentType.RECIPE: "Structure recipe with ingredients and instructions",
            ContentType.FINANCIAL_DATA: "Process transactions and update financial records",
            ContentType.LOG_ENTRIES: "Parse log entries for monitoring and analysis",
            ContentType.EMPLOYEE_DATA: "Update employee database with HR information",
            ContentType.EMAIL_THREAD: "Extract email metadata and conversation flow",
            ContentType.MEETING_NOTES: "Parse meeting details and action items",
            ContentType.UNKNOWN: "Store for manual review and classification"
        }
        
        base_action = actions.get(content_type, "Process as generic content")
        return f"{action_prefix} {base_action}"
