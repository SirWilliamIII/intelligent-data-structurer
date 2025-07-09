"""
Business Document Classifier

Specialized classifier for common business document types with high accuracy.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

from core.classifier import IntelligentClassifier, ClassificationResult, ContentType
from core.business_document_types import (
    BusinessDocumentType, 
    BUSINESS_DOCUMENT_PATTERNS,
    get_collection_name,
    get_document_category
)

@dataclass
class BusinessClassificationResult:
    """Extended classification result for business documents."""
    business_type: Optional[BusinessDocumentType]
    content_type: ContentType  # Fallback to general type
    confidence: float
    reasoning: str
    suggested_collection: str
    category: str
    metadata: Dict[str, any]
    alternative_types: List[Tuple[BusinessDocumentType, float]]

class BusinessDocumentClassifier(IntelligentClassifier):
    """Specialized classifier for business documents."""
    
    def __init__(self):
        super().__init__()
        self.business_patterns = BUSINESS_DOCUMENT_PATTERNS
        
    async def classify_business_document(self, content: str, filename: str = "") -> BusinessClassificationResult:
        """Classify content as a business document with detailed analysis."""
        
        # First try business-specific classification
        business_result = self._classify_business_content(content, filename)
        
        # If confidence is low, fall back to general classification
        if business_result.confidence < 0.5:
            general_result = await self.classify_content(content, filename)
            
            # Map general types to business types if possible
            business_type = self._map_general_to_business(general_result.content_type)
            
            if business_type:
                return BusinessClassificationResult(
                    business_type=business_type,
                    content_type=general_result.content_type,
                    confidence=general_result.confidence * 0.8,  # Reduce confidence for mapping
                    reasoning=f"Mapped from general type: {general_result.reasoning}",
                    suggested_collection=get_collection_name(business_type),
                    category=get_document_category(business_type),
                    metadata=self._extract_business_metadata(content, business_type),
                    alternative_types=[]
                )
        
        return business_result
    
    def _classify_business_content(self, content: str, filename: str) -> BusinessClassificationResult:
        """Classify content using business-specific patterns."""
        
        content_lower = content.lower()
        filename_lower = filename.lower()
        scores = {}
        metadata_extracted = {}
        
        # Check each business document type
        for doc_type, pattern in self.business_patterns.items():
            score = 0
            matches = []
            metadata = {}
            
            # Check keywords (weighted: 1 point each)
            keyword_matches = sum(1 for keyword in pattern.keywords 
                                if keyword in content_lower or keyword in filename_lower)
            score += keyword_matches
            matches.extend([f"keyword '{kw}'" for kw in pattern.keywords 
                          if kw in content_lower or kw in filename_lower][:3])
            
            # Check required sections (weighted: 3 points each)
            section_matches = 0
            for section in pattern.required_sections:
                if self._has_section(content_lower, section):
                    section_matches += 1
                    matches.append(f"section '{section}'")
            score += section_matches * 3
            
            # Check common phrases (weighted: 2 points each)
            phrase_matches = sum(1 for phrase in pattern.common_phrases 
                               if phrase in content_lower)
            score += phrase_matches * 2
            
            # Check regex patterns (weighted: 4 points each)
            pattern_matches = 0
            for regex in pattern.regex_patterns:
                match = re.search(regex, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    pattern_matches += 1
                    matches.append("pattern match")
                    # Extract metadata from regex matches
                    if 'invoice' in regex and match:
                        metadata['invoice_number'] = match.group(0)
                    elif 'date' in regex and match:
                        metadata['date'] = match.group(0)
                    elif 'total' in regex and match:
                        metadata['total'] = match.group(0)
            score += pattern_matches * 4
            
            # Check document length
            word_count = len(content.split())
            min_words, max_words = pattern.typical_length_range
            if min_words <= word_count <= max_words:
                score += 2
                matches.append("typical length")
            
            # Normalize score
            max_possible = (len(pattern.keywords) + 
                          len(pattern.required_sections) * 3 + 
                          len(pattern.common_phrases) * 2 + 
                          len(pattern.regex_patterns) * 4 + 2)
            
            confidence = min(score / max_possible, 1.0) if max_possible > 0 else 0.0
            
            # Apply confidence threshold
            if confidence >= pattern.confidence_threshold:
                scores[doc_type] = (confidence, matches, metadata)
        
        # Select best match
        if scores:
            sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            best_type, (confidence, matches, metadata) = sorted_scores[0]
            
            # Get alternatives
            alternatives = [(doc_type, score[0]) for doc_type, score in sorted_scores[1:4]]
            
            # Extract additional metadata
            full_metadata = self._extract_business_metadata(content, best_type)
            full_metadata.update(metadata)
            
            reasoning = f"Business document analysis: {', '.join(matches[:3])}"
            
            return BusinessClassificationResult(
                business_type=best_type,
                content_type=self._get_content_type_mapping(best_type),
                confidence=confidence,
                reasoning=reasoning,
                suggested_collection=get_collection_name(best_type),
                category=get_document_category(best_type),
                metadata=full_metadata,
                alternative_types=alternatives
            )
        
        # No business type matched
        return BusinessClassificationResult(
            business_type=None,
            content_type=ContentType.UNKNOWN,
            confidence=0.0,
            reasoning="No business document pattern matched",
            suggested_collection="business_general",
            category="General Business",
            metadata={},
            alternative_types=[]
        )
    
    def _has_section(self, content: str, section: str) -> bool:
        """Check if content has a specific section."""
        section_indicators = {
            'bill_to': ['bill to', 'billing address', 'billed to'],
            'ship_to': ['ship to', 'shipping address', 'deliver to'],
            'items': ['item', 'description', 'quantity', 'price'],
            'total': ['total', 'amount due', 'grand total'],
            'contact': ['email', 'phone', 'address'],
            'experience': ['experience', 'employment', 'work history'],
            'education': ['education', 'degree', 'university', 'school'],
            'from': ['from:', 'sender:'],
            'to': ['to:', 'recipient:'],
            'subject': ['subject:', 're:'],
            'parties': ['party', 'parties', 'agreement between'],
            'terms': ['terms', 'conditions', 'obligations'],
            'signatures': ['signature', 'signed', 'authorized by'],
            'attendees': ['attendees', 'present', 'participants'],
            'agenda': ['agenda', 'topics', 'discussion points'],
            'discussion': ['discussion', 'minutes', 'notes'],
            'vendor': ['vendor', 'supplier', 'sold by'],
            'employee': ['employee', 'name', 'id'],
            'expenses': ['expense', 'cost', 'amount'],
            'objectives': ['objective', 'goal', 'purpose'],
            'timeline': ['timeline', 'schedule', 'deadline'],
            'deliverables': ['deliverable', 'output', 'milestone'],
            'customer': ['customer', 'client', 'bill to'],
            'validity': ['valid', 'expires', 'quotation date'],
            'period': ['period', 'week ending', 'dates'],
            'hours': ['hours', 'time', 'overtime']
        }
        
        indicators = section_indicators.get(section, [section])
        return any(indicator in content for indicator in indicators)
    
    def _extract_business_metadata(self, content: str, doc_type: BusinessDocumentType) -> Dict[str, any]:
        """Extract business-specific metadata from content."""
        metadata = {
            'extracted_at': datetime.now().isoformat(),
            'document_type': doc_type.value
        }
        
        # Extract common business entities
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        if emails:
            metadata['emails'] = list(set(emails))
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, content)
        if dates:
            metadata['dates'] = dates
        
        # Extract money amounts
        money_pattern = r'\$\s*[\d,]+\.?\d*'
        amounts = re.findall(money_pattern, content)
        if amounts:
            metadata['amounts'] = amounts
        
        # Document-specific extraction
        if doc_type == BusinessDocumentType.INVOICE:
            # Extract invoice-specific data
            invoice_num = re.search(r'invoice\s*#?\s*:?\s*(\w+)', content, re.IGNORECASE)
            if invoice_num:
                metadata['invoice_number'] = invoice_num.group(1)
                
            due_date = re.search(r'due\s+date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', content, re.IGNORECASE)
            if due_date:
                metadata['due_date'] = due_date.group(1)
        
        elif doc_type == BusinessDocumentType.RESUME:
            # Extract resume-specific data
            phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            phones = re.findall(phone_pattern, content)
            if phones:
                metadata['phone_numbers'] = phones
            
            # Extract years of experience
            exp_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?experience'
            exp_match = re.search(exp_pattern, content, re.IGNORECASE)
            if exp_match:
                metadata['years_experience'] = int(exp_match.group(1))
        
        return metadata
    
    def _map_general_to_business(self, content_type: ContentType) -> Optional[BusinessDocumentType]:
        """Map general content types to business document types."""
        mapping = {
            ContentType.INVOICE: BusinessDocumentType.INVOICE,
            ContentType.EMAIL_THREAD: BusinessDocumentType.BUSINESS_EMAIL,
            ContentType.MEETING_NOTES: BusinessDocumentType.MEETING_MINUTES,
            ContentType.RESUME_CV: BusinessDocumentType.RESUME,
            ContentType.FINANCIAL_DATA: BusinessDocumentType.FINANCIAL_STATEMENT,
            ContentType.EMPLOYEE_DATA: BusinessDocumentType.EMPLOYEE_CONTRACT
        }
        return mapping.get(content_type)
    
    def _get_content_type_mapping(self, business_type: BusinessDocumentType) -> ContentType:
        """Map business types back to general content types."""
        mapping = {
            BusinessDocumentType.INVOICE: ContentType.INVOICE,
            BusinessDocumentType.BUSINESS_EMAIL: ContentType.EMAIL_THREAD,
            BusinessDocumentType.MEETING_MINUTES: ContentType.MEETING_NOTES,
            BusinessDocumentType.RESUME: ContentType.RESUME_CV,
            BusinessDocumentType.FINANCIAL_STATEMENT: ContentType.FINANCIAL_DATA,
            BusinessDocumentType.EMPLOYEE_CONTRACT: ContentType.EMPLOYEE_DATA,
            BusinessDocumentType.PURCHASE_ORDER: ContentType.INVOICE,
            BusinessDocumentType.EXPENSE_REPORT: ContentType.FINANCIAL_DATA,
            BusinessDocumentType.CONTRACT: ContentType.UNKNOWN,
            BusinessDocumentType.PROJECT_PLAN: ContentType.UNKNOWN,
            BusinessDocumentType.TIMESHEET: ContentType.EMPLOYEE_DATA
        }
        return mapping.get(business_type, ContentType.UNKNOWN)