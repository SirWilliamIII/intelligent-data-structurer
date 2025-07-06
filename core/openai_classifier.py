"""
OpenAI-powered content classification for high accuracy.
"""

import asyncio
from openai import AsyncOpenAI
from typing import Dict, Any, Optional
from loguru import logger
from .config import settings
from .classifier import ContentType, ClassificationResult

class OpenAIClassifier:
    """High-accuracy content classifier using OpenAI."""
    
    def __init__(self):
        self.client = None
        if settings.openai_api_key and settings.use_openai_classification:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("OpenAI classifier initialized")
        else:
            logger.warning("OpenAI classifier disabled - no API key or disabled in settings")
    
    def is_available(self) -> bool:
        """Check if OpenAI classification is available."""
        return self.client is not None
    
    async def classify_content(self, content: str, filename: str = "") -> Optional[ClassificationResult]:
        """Classify content using OpenAI with high accuracy."""
        
        if not self.is_available():
            return None
        
        try:
            # Create comprehensive prompt for classification
            prompt = self._create_classification_prompt(content, filename)
            
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            # Parse response
            result = self._parse_openai_response(response.choices[0].message.content)
            
            logger.info(f"OpenAI classified as {result.content_type.value} with {result.confidence:.2f} confidence")
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI classification failed: {e}")
            return None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for content classification."""
        
        content_types = [ct.value for ct in ContentType if ct != ContentType.UNKNOWN]
        
        return f"""You are an expert content classifier. Analyze the provided text and classify it into one of these categories:

{', '.join(content_types)}

For each classification, provide:
1. The most likely content type
2. Confidence score (0.0 to 1.0)
3. Brief reasoning (1-2 sentences)
4. Suggested action

Content Types Explained:
- contact_info: Personal/business contact details, names, emails, phones, addresses
- business_card: Formatted business card information with titles and companies
- product_data: Product catalogs, inventory, pricing, SKUs, specifications
- invoice: Bills, invoices, payment information, line items, totals
- event_info: Events, meetings, conferences, schedules, dates, locations
- article: News articles, blog posts, written content with titles/authors
- recipe: Cooking recipes with ingredients, instructions, prep times
- financial_data: Financial transactions, bank statements, accounting data
- log_entries: System logs, error logs, application logs with timestamps
- employee_data: HR information, employee records, personnel data
- email_thread: Email conversations, replies, forward chains
- meeting_notes: Meeting minutes, notes, action items, attendees
- cover_letter: Job application letters, personal statements
- resume_cv: Resumes, CVs, professional experience summaries
- personal_statement: Personal essays, autobiographical content
- unknown: Content that doesn't fit other categories

Respond in this exact JSON format:
{
  "content_type": "category_name",
  "confidence": 0.95,
  "reasoning": "Brief explanation of why this classification was chosen",
  "suggested_action": "What should be done with this content"
}"""

    def _create_classification_prompt(self, content: str, filename: str) -> str:
        """Create the classification prompt."""
        
        # Truncate very long content
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "\n... [content truncated]"
        
        prompt = f"""Please classify this content:

FILENAME: {filename}

CONTENT:
{content}

Analyze the structure, language patterns, intent, and context to determine the most appropriate category. Consider the filename as additional context."""

        return prompt
    
    def _parse_openai_response(self, response_text: str) -> ClassificationResult:
        """Parse OpenAI response into ClassificationResult."""
        
        try:
            import json
            import re
            
            # Extract JSON from response (handle any extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group())
            
            # Validate content type
            content_type_str = data.get("content_type", "unknown")
            try:
                content_type = ContentType(content_type_str)
            except ValueError:
                logger.warning(f"Unknown content type from OpenAI: {content_type_str}")
                content_type = ContentType.UNKNOWN
            
            # Extract other fields
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "OpenAI classification")
            suggested_action = data.get("suggested_action", "Process based on classification")
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return ClassificationResult(
                content_type=content_type,
                confidence=confidence,
                reasoning=f"OpenAI: {reasoning}",
                suggested_action=f"Auto-process: {suggested_action}" if confidence >= 0.8 else f"Review: {suggested_action}",
                alternative_types=[]  # OpenAI gives us the best classification
            )
            
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Return fallback result
            return ClassificationResult(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"OpenAI parsing failed: {str(e)}",
                suggested_action="Manual classification needed",
                alternative_types=[]
            )
