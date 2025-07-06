"""
Enhanced NLP analyzer for sentiment-based dynamic table creation.
"""

import re
import json
import spacy
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from loguru import logger
from enum import Enum

@dataclass
class AnalysisResult:
    """Enhanced analysis result with sentiment and priority."""
    table_name: str
    sentiment_score: float  # -1 to 1
    urgency_score: float    # 0 to 10
    importance_score: float # 0 to 10
    risk_score: float      # 0 to 10
    emotions: List[str]
    entities: List[Dict[str, Any]]
    topics: List[str]
    confidence: float
    reasoning: str
    extracted_data: Dict[str, Any]

class SentimentCategory(Enum):
    URGENT_ISSUES = "urgent_issues"
    COMPLAINTS = "complaints"
    POSITIVE_FEEDBACK = "positive_feedback"
    POLICY_VIOLATIONS = "policy_violations"
    SENSITIVE_DATA = "sensitive_data"
    ESCALATION_NEEDED = "escalation_needed"
    CUSTOMER_FEEDBACK = "customer_feedback"
    INTERNAL_COMMUNICATIONS = "internal_communications"
    COMPLIANCE_ISSUES = "compliance_issues"
    HIGH_PRIORITY_ITEMS = "high_priority_items"
    ROUTINE_DOCUMENTS = "routine_documents"
    ARCHIVED_CONTENT = "archived_content"

class EnhancedAnalyzer:
    """Enhanced analyzer with sentiment-based table creation."""
    
    def __init__(self):
        self.nlp = None
        self._load_models()
        
        # Sentiment keywords
        self.urgent_keywords = [
            'urgent', 'emergency', 'immediate', 'asap', 'critical', 'crisis',
            'deadline', 'overdue', 'escalate', 'priority', 'rush', 'time-sensitive'
        ]
        
        self.complaint_keywords = [
            'complaint', 'dissatisfied', 'unhappy', 'disappointed', 'frustrated',
            'angry', 'unacceptable', 'poor service', 'issue', 'problem'
        ]
        
        self.positive_keywords = [
            'excellent', 'outstanding', 'satisfied', 'happy', 'pleased',
            'great', 'wonderful', 'fantastic', 'appreciate', 'thank you'
        ]
        
        self.policy_keywords = [
            'violation', 'breach', 'non-compliance', 'policy', 'regulation',
            'audit', 'investigation', 'misconduct', 'unauthorized'
        ]
        
        self.sensitive_keywords = [
            'confidential', 'classified', 'restricted', 'proprietary', 'private',
            'ssn', 'social security', 'credit card', 'password', 'personal'
        ]
        
    def _load_models(self):
        """Load NLP models."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Enhanced NLP models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLP models: {e}")
            self.nlp = None
    
    async def analyze_content(self, content: str, filename: str) -> AnalysisResult:
        """Perform enhanced analysis to determine table and priority."""
        
        if not self.nlp:
            return self._fallback_analysis(content, filename)
        
        # Process with spaCy
        doc = self.nlp(content)
        
        # Extract entities
        entities = [
            {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_) or ent.label_
            }
            for ent in doc.ents
        ]
        
        # Sentiment analysis
        sentiment_score = self._analyze_sentiment(content)
        
        # Urgency detection
        urgency_score = self._calculate_urgency(content)
        
        # Importance scoring
        importance_score = self._calculate_importance(content, entities)
        
        # Risk assessment
        risk_score = self._assess_risk(content, entities)
        
        # Emotion detection
        emotions = self._detect_emotions(content)
        
        # Topic extraction
        topics = self._extract_topics(content, entities)
        
        # Determine table based on analysis
        table_name, confidence, reasoning = self._determine_table(
            sentiment_score, urgency_score, importance_score, 
            risk_score, emotions, topics, content
        )
        
        # Extract structured data
        extracted_data = self._extract_structured_data(content, entities, filename)
        
        return AnalysisResult(
            table_name=table_name,
            sentiment_score=sentiment_score,
            urgency_score=urgency_score,
            importance_score=importance_score,
            risk_score=risk_score,
            emotions=emotions,
            entities=entities,
            topics=topics,
            confidence=confidence,
            reasoning=reasoning,
            extracted_data=extracted_data
        )
    
    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment (-1 to 1)."""
        content_lower = content.lower()
        
        # Count positive/negative indicators
        positive_count = sum(1 for word in self.positive_keywords if word in content_lower)
        negative_count = sum(1 for word in self.complaint_keywords if word in content_lower)
        
        # Simple sentiment scoring
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _calculate_urgency(self, content: str) -> float:
        """Calculate urgency score (0-10)."""
        content_lower = content.lower()
        
        urgency_indicators = 0
        for keyword in self.urgent_keywords:
            if keyword in content_lower:
                urgency_indicators += content_lower.count(keyword)
        
        # Check for time-sensitive patterns
        if re.search(r'\b(today|tomorrow|this week|deadline|due|overdue)\b', content_lower):
            urgency_indicators += 2
        
        # Check for exclamation marks and caps
        if content.count('!') > 2:
            urgency_indicators += 1
        if len(re.findall(r'\b[A-Z]{3,}\b', content)) > 2:
            urgency_indicators += 1
        
        return min(urgency_indicators, 10)
    
    def _calculate_importance(self, content: str, entities: List[Dict]) -> float:
        """Calculate importance score (0-10)."""
        importance = 0
        
        # Check for important entities
        important_labels = ['PERSON', 'ORG', 'MONEY', 'DATE', 'GPE']
        for entity in entities:
            if entity['label'] in important_labels:
                importance += 0.5
        
        # Check for financial/legal terms
        if re.search(r'\$[\d,]+|contract|agreement|legal|lawsuit|payment', content.lower()):
            importance += 2
        
        # Check for executive/management terms
        if re.search(r'\b(ceo|president|director|manager|executive|board)\b', content.lower()):
            importance += 1
        
        return min(importance, 10)
    
    def _assess_risk(self, content: str, entities: List[Dict]) -> float:
        """Assess risk level (0-10)."""
        risk = 0
        content_lower = content.lower()
        
        # Check for sensitive data
        for keyword in self.sensitive_keywords:
            if keyword in content_lower:
                risk += 1
        
        # Check for policy violations
        for keyword in self.policy_keywords:
            if keyword in content_lower:
                risk += 1.5
        
        # Check for PII patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', content):  # SSN pattern
            risk += 3
        if re.search(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', content):  # Credit card
            risk += 3
        
        return min(risk, 10)
    
    def _detect_emotions(self, content: str) -> List[str]:
        """Detect emotions in content."""
        emotions = []
        content_lower = content.lower()
        
        emotion_patterns = {
            'anger': ['angry', 'furious', 'outraged', 'mad', 'irritated'],
            'fear': ['worried', 'concerned', 'anxious', 'scared', 'afraid'],
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'thrilled'],
            'sadness': ['sad', 'disappointed', 'upset', 'frustrated', 'discouraged'],
            'surprise': ['surprised', 'amazed', 'shocked', 'unexpected', 'stunned']
        }
        
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                emotions.append(emotion)
        
        return emotions
    
    def _extract_topics(self, content: str, entities: List[Dict]) -> List[str]:
        """Extract topics from content."""
        topics = []
        content_lower = content.lower()
        
        # Topic categories
        topic_patterns = {
            'hr': ['employee', 'hiring', 'payroll', 'benefits', 'performance'],
            'finance': ['budget', 'revenue', 'cost', 'profit', 'invoice'],
            'legal': ['contract', 'agreement', 'compliance', 'regulation', 'audit'],
            'customer_service': ['customer', 'client', 'support', 'service', 'feedback'],
            'operations': ['process', 'workflow', 'procedure', 'system', 'operation'],
            'technology': ['software', 'system', 'data', 'security', 'technical']
        }
        
        for topic, keywords in topic_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _determine_table(self, sentiment: float, urgency: float, importance: float, 
                        risk: float, emotions: List[str], topics: List[str], 
                        content: str) -> Tuple[str, float, str]:
        """Determine table name based on analysis."""
        
        reasoning_parts = []
        
        # High risk content
        if risk >= 7:
            reasoning_parts.append(f"High risk score ({risk:.1f})")
            if 'policy' in content.lower() or 'violation' in content.lower():
                return SentimentCategory.POLICY_VIOLATIONS.value, 0.9, f"Policy violation detected. {' '.join(reasoning_parts)}"
            return SentimentCategory.SENSITIVE_DATA.value, 0.85, f"Sensitive data detected. {' '.join(reasoning_parts)}"
        
        # Urgent issues
        if urgency >= 7:
            reasoning_parts.append(f"High urgency score ({urgency:.1f})")
            return SentimentCategory.URGENT_ISSUES.value, 0.9, f"Urgent issue detected. {' '.join(reasoning_parts)}"
        
        # Escalation needed
        if urgency >= 5 and sentiment <= -0.5:
            reasoning_parts.append(f"High urgency ({urgency:.1f}) with negative sentiment ({sentiment:.1f})")
            return SentimentCategory.ESCALATION_NEEDED.value, 0.8, f"Escalation needed. {' '.join(reasoning_parts)}"
        
        # Complaints
        if sentiment <= -0.3:
            reasoning_parts.append(f"Negative sentiment ({sentiment:.1f})")
            return SentimentCategory.COMPLAINTS.value, 0.75, f"Complaint detected. {' '.join(reasoning_parts)}"
        
        # Positive feedback
        if sentiment >= 0.3:
            reasoning_parts.append(f"Positive sentiment ({sentiment:.1f})")
            return SentimentCategory.POSITIVE_FEEDBACK.value, 0.75, f"Positive feedback detected. {' '.join(reasoning_parts)}"
        
        # Compliance issues
        if risk >= 4 and 'compliance' in topics:
            reasoning_parts.append(f"Compliance-related content with risk score ({risk:.1f})")
            return SentimentCategory.COMPLIANCE_ISSUES.value, 0.7, f"Compliance issue detected. {' '.join(reasoning_parts)}"
        
        # High priority items
        if importance >= 6:
            reasoning_parts.append(f"High importance score ({importance:.1f})")
            return SentimentCategory.HIGH_PRIORITY_ITEMS.value, 0.7, f"High priority item detected. {' '.join(reasoning_parts)}"
        
        # Customer feedback
        if 'customer_service' in topics:
            reasoning_parts.append("Customer service related")
            return SentimentCategory.CUSTOMER_FEEDBACK.value, 0.65, f"Customer feedback detected. {' '.join(reasoning_parts)}"
        
        # Internal communications
        if urgency < 3 and importance < 4:
            reasoning_parts.append(f"Low urgency ({urgency:.1f}) and importance ({importance:.1f})")
            return SentimentCategory.ROUTINE_DOCUMENTS.value, 0.6, f"Routine document. {' '.join(reasoning_parts)}"
        
        # Default to internal communications
        return SentimentCategory.INTERNAL_COMMUNICATIONS.value, 0.5, f"Internal communication. {' '.join(reasoning_parts)}"
    
    def _extract_structured_data(self, content: str, entities: List[Dict], filename: str) -> Dict[str, Any]:
        """Extract structured data with enhanced fields."""
        
        # Extract emails, phones, dates
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        phones = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', content)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b', content)
        
        return {
            'content': content,
            'entities': entities,
            'emails': emails,
            'phones': phones,
            'dates': dates,
            'word_count': len(content.split()),
            'char_count': len(content),
            'source_file': filename,
            'extracted_at': datetime.utcnow().isoformat()
        }
    
    def _fallback_analysis(self, content: str, filename: str) -> AnalysisResult:
        """Fallback analysis when NLP models aren't available."""
        return AnalysisResult(
            table_name=SentimentCategory.ROUTINE_DOCUMENTS.value,
            sentiment_score=0.0,
            urgency_score=0.0,
            importance_score=0.0,
            risk_score=0.0,
            emotions=[],
            entities=[],
            topics=[],
            confidence=0.3,
            reasoning="Fallback analysis - NLP models not available",
            extracted_data=self._extract_structured_data(content, [], filename)
        )