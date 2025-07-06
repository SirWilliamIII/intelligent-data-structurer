"""
Data extractors for converting unstructured content into structured data.
"""

import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from .classifier import ContentType

class DataExtractor:
    """Extract structured data from unstructured text based on content type."""
    
    def __init__(self):
        # Common regex patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
        self.price_pattern = re.compile(r'\$\d+(?:\.\d{2})?')
        self.date_pattern = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b')
        self.time_pattern = re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b')
        
    async def extract_data(self, content: str, content_type: ContentType, filename: str) -> Dict[str, Any]:
        """Extract structured data based on content type."""
        
        extractors = {
            ContentType.CONTACT_INFO: self._extract_contact_info,
            ContentType.BUSINESS_CARD: self._extract_business_card,
            ContentType.PRODUCT_DATA: self._extract_product_data,
            ContentType.EVENT_INFO: self._extract_event_info,
            ContentType.RECIPE: self._extract_recipe,
            ContentType.FINANCIAL_DATA: self._extract_financial_data,
            ContentType.EMPLOYEE_DATA: self._extract_employee_data,
            ContentType.ARTICLE: self._extract_article,
            ContentType.COVER_LETTER: self._extract_cover_letter,
            ContentType.RESUME_CV: self._extract_resume,
        }
        
        extractor = extractors.get(content_type, self._extract_generic)
        
        try:
            data = await extractor(content)
            data['source_file'] = filename
            data['extracted_at'] = datetime.utcnow().isoformat()
            return data
            
        except Exception as e:
            logger.error(f"Extraction failed for {content_type}: {e}")
            return self._extract_generic(content)
    
    async def _extract_contact_info(self, content: str) -> Dict[str, Any]:
        """Extract contact information."""
        
        emails = self.email_pattern.findall(content)
        phones = self.phone_pattern.findall(content)
        
        # Extract name (look for patterns like "Name: John Smith")
        name_match = re.search(r'(?:name|contact):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', content, re.IGNORECASE)
        name = name_match.group(1) if name_match else None
        
        # Extract company
        company_match = re.search(r'(?:company|corp|inc|llc):\s*(.+)', content, re.IGNORECASE)
        company = company_match.group(1).strip() if company_match else None
        
        # Extract title
        title_match = re.search(r'(?:title|position):\s*(.+)', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else None
        
        # Extract address (simple heuristic)
        address_match = re.search(r'(?:address):\s*(.+(?:\n.+)*)', content, re.IGNORECASE | re.MULTILINE)
        address = address_match.group(1).strip() if address_match else None
        
        return {
            'name': name,
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None,
            'mobile': phones[1] if len(phones) > 1 else None,
            'company': company,
            'title': title,
            'address': address,
            'notes': content[:500] + '...' if len(content) > 500 else content
        }
    
    async def _extract_business_card(self, content: str) -> Dict[str, Any]:
        """Extract business card information."""
        
        emails = self.email_pattern.findall(content)
        phones = self.phone_pattern.findall(content)
        
        # Business card specific patterns
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        name = None
        title = None
        company = None
        website = None
        
        # Heuristics for business card parsing
        for i, line in enumerate(lines):
            # First non-empty line often contains name
            if not name and re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line):
                name = line
            
            # Look for titles
            if any(keyword in line.lower() for keyword in ['ceo', 'cto', 'manager', 'director', 'president']):
                title = line
            
            # Look for company names
            if any(keyword in line.lower() for keyword in ['corp', 'inc', 'llc', 'ltd', 'company']):
                company = line
            
            # Look for websites
            if 'www.' in line.lower() or '.com' in line.lower():
                website = line
        
        return {
            'name': name,
            'title': title,
            'company': company,
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None,
            'website': website,
            'address': None  # Could be extracted with more sophisticated parsing
        }
    
    async def _extract_product_data(self, content: str) -> Dict[str, Any]:
        """Extract product information."""
        
        products = []
        
        # Split content into product sections
        sections = re.split(r'\n\s*\n|\n(?=Product:)', content)
        
        for section in sections:
            if not section.strip():
                continue
                
            product = {}
            
            # Extract product name
            name_match = re.search(r'(?:Product|Item):\s*(.+)', section, re.IGNORECASE)
            if name_match:
                product['name'] = name_match.group(1).strip()
            
            # Extract SKU
            sku_match = re.search(r'SKU:\s*(\S+)', section, re.IGNORECASE)
            if sku_match:
                product['sku'] = sku_match.group(1)
            
            # Extract price
            price_matches = self.price_pattern.findall(section)
            if price_matches:
                price_str = price_matches[0].replace('$', '')
                try:
                    product['price'] = float(price_str)
                except ValueError:
                    product['price'] = None
            
            # Extract category
            category_match = re.search(r'Category:\s*(.+)', section, re.IGNORECASE)
            if category_match:
                product['category'] = category_match.group(1).strip()
            
            # Extract description
            desc_match = re.search(r'Description:\s*(.+)', section, re.IGNORECASE)
            if desc_match:
                product['description'] = desc_match.group(1).strip()
            
            # Extract stock
            stock_match = re.search(r'(?:In Stock|Stock):\s*(\d+)', section, re.IGNORECASE)
            if stock_match:
                product['in_stock'] = int(stock_match.group(1))
            
            # Extract supplier
            supplier_match = re.search(r'Supplier:\s*(.+)', section, re.IGNORECASE)
            if supplier_match:
                product['supplier'] = supplier_match.group(1).strip()
            
            if product:  # Only add if we extracted something
                products.append(product)
        
        # Return the first product or a summary
        if products:
            return products[0]  # For now, return first product
        else:
            return {'name': 'Unknown Product', 'description': content[:200]}
    
    async def _extract_event_info(self, content: str) -> Dict[str, Any]:
        """Extract event information."""
        
        # Extract title
        title_match = re.search(r'(?:Event|Conference|Meeting):\s*(.+)', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else None
        
        # Extract dates
        dates = self.date_pattern.findall(content)
        event_date = dates[0] if dates else None
        
        # Extract times
        times = self.time_pattern.findall(content)
        event_time = times[0] if times else None
        
        # Extract location
        location_match = re.search(r'(?:Location|Venue):\s*(.+)', content, re.IGNORECASE)
        location = location_match.group(1).strip() if location_match else None
        
        # Extract organizer
        org_match = re.search(r'Organizer:\s*(.+)', content, re.IGNORECASE)
        organizer = org_match.group(1).strip() if org_match else None
        
        # Extract capacity
        capacity_match = re.search(r'(?:Capacity|Attendees):\s*(\d+)', content, re.IGNORECASE)
        capacity = int(capacity_match.group(1)) if capacity_match else None
        
        return {
            'title': title,
            'event_date': event_date,
            'event_time': event_time,
            'location': location,
            'organizer': organizer,
            'capacity': capacity,
            'description': content[:500] + '...' if len(content) > 500 else content
        }
    
    async def _extract_recipe(self, content: str) -> Dict[str, Any]:
        """Extract recipe information."""
        
        # Extract title (often the first line or after #)
        title_match = re.search(r'^#\s*(.+)|^(.+)\n', content, re.MULTILINE)
        title = None
        if title_match:
            title = title_match.group(1) or title_match.group(2)
            if title:
                title = title.strip()
        
        # Extract prep time
        prep_match = re.search(r'Prep\s+Time:\s*(.+)', content, re.IGNORECASE)
        prep_time = prep_match.group(1).strip() if prep_match else None
        
        # Extract cook time
        cook_match = re.search(r'Cook\s+Time:\s*(.+)', content, re.IGNORECASE)
        cook_time = cook_match.group(1).strip() if cook_match else None
        
        # Extract servings
        serving_match = re.search(r'Servings?:\s*(\d+)', content, re.IGNORECASE)
        servings = int(serving_match.group(1)) if serving_match else None
        
        # Extract ingredients
        ingredients = []
        ingredients_section = re.search(r'Ingredients?:?\s*\n(.*?)(?=\n\s*(?:Instructions?|Directions?|Steps?)|\Z)', 
                                      content, re.IGNORECASE | re.DOTALL)
        if ingredients_section:
            ingredient_text = ingredients_section.group(1)
            # Split by lines and clean up
            for line in ingredient_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+', line)):
                    # Clean up bullet points and numbers
                    ingredient = re.sub(r'^[-•*]\s*|\d+\.\s*', '', line).strip()
                    if ingredient:
                        ingredients.append(ingredient)
        
        # Extract instructions
        instructions = []
        instructions_section = re.search(r'(?:Instructions?|Directions?|Steps?):?\s*\n(.*)', 
                                       content, re.IGNORECASE | re.DOTALL)
        if instructions_section:
            instruction_text = instructions_section.group(1)
            # Split by lines and clean up
            for line in instruction_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+', line)):
                    # Clean up bullet points and numbers
                    instruction = re.sub(r'^[-•*]\s*|\d+\.\s*', '', line).strip()
                    if instruction:
                        instructions.append(instruction)
        
        return {
            'title': title,
            'prep_time': prep_time,
            'cook_time': cook_time,
            'servings': servings,
            'ingredients': ingredients,
            'instructions': instructions
        }
    
    async def _extract_financial_data(self, content: str) -> Dict[str, Any]:
        """Extract financial transaction data."""
        
        # Extract transaction ID
        trans_match = re.search(r'(?:Transaction|Trans)\s*(?:ID|#):\s*(\S+)', content, re.IGNORECASE)
        transaction_id = trans_match.group(1) if trans_match else None
        
        # Extract amount
        amounts = self.price_pattern.findall(content)
        amount = None
        if amounts:
            amount_str = amounts[0].replace('$', '')
            try:
                amount = float(amount_str)
            except ValueError:
                pass
        
        # Extract date
        dates = self.date_pattern.findall(content)
        date = dates[0] if dates else None
        
        # Extract account
        account_match = re.search(r'Account(?:\s+Number)?:\s*(\S+)', content, re.IGNORECASE)
        account = account_match.group(1) if account_match else None
        
        # Extract type
        type_match = re.search(r'(?:Type|Transaction Type):\s*(.+)', content, re.IGNORECASE)
        trans_type = type_match.group(1).strip() if type_match else None
        
        return {
            'transaction_id': transaction_id,
            'date': date,
            'amount': amount,
            'account': account,
            'type': trans_type,
            'description': content[:200] + '...' if len(content) > 200 else content
        }
    
    async def _extract_employee_data(self, content: str) -> Dict[str, Any]:
        """Extract employee information."""
        
        # Extract employee ID
        id_match = re.search(r'(?:Employee|ID):\s*(\S+)', content, re.IGNORECASE)
        employee_id = id_match.group(1) if id_match else None
        
        # Extract name
        name_match = re.search(r'Name:\s*(.+)', content, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else None
        
        # Extract email
        emails = self.email_pattern.findall(content)
        email = emails[0] if emails else None
        
        # Extract department
        dept_match = re.search(r'Department:\s*(.+)', content, re.IGNORECASE)
        department = dept_match.group(1).strip() if dept_match else None
        
        # Extract position
        pos_match = re.search(r'(?:Position|Title):\s*(.+)', content, re.IGNORECASE)
        position = pos_match.group(1).strip() if pos_match else None
        
        # Extract salary
        salary_match = re.search(r'Salary:\s*\$?([\d,]+)', content, re.IGNORECASE)
        salary = None
        if salary_match:
            try:
                salary = float(salary_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        return {
            'employee_id': employee_id,
            'name': name,
            'email': email,
            'department': department,
            'position': position,
            'salary': salary
        }
    
    async def _extract_article(self, content: str) -> Dict[str, Any]:
        """Extract article metadata and content."""
        
        lines = content.split('\n')
        
        # Extract title (first non-empty line or markdown header)
        title = None
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('#'):
                    title = line.lstrip('#').strip()
                    break
                elif not title:
                    title = line
                    break
        
        # Extract author
        author_match = re.search(r'(?:by|author):\s*(.+)|by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', content, re.IGNORECASE)
        author = None
        if author_match:
            author = (author_match.group(1) or author_match.group(2)).strip()
        
        # Extract publish date
        dates = self.date_pattern.findall(content)
        publish_date = dates[0] if dates else None
        
        # Word count
        word_count = len(content.split())
        
        # Extract summary (first paragraph)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        summary = paragraphs[1] if len(paragraphs) > 1 else paragraphs[0] if paragraphs else None
        if summary and len(summary) > 200:
            summary = summary[:200] + '...'
        
        return {
            'title': title,
            'author': author,
            'publish_date': publish_date,
            'content': content,
            'summary': summary,
            'word_count': word_count,
            'tags': []  # Could be extracted with NLP
        }
    
    async def _extract_cover_letter(self, content: str) -> Dict[str, Any]:
        """Extract cover letter information."""
        return await self._extract_generic(content)
    
    async def _extract_resume(self, content: str) -> Dict[str, Any]:
        """Extract resume/CV information."""
        return await self._extract_generic(content)
    
    async def _extract_generic(self, content: str) -> Dict[str, Any]:
        """Generic extraction for unknown content types."""
        
        emails = self.email_pattern.findall(content)
        phones = self.phone_pattern.findall(content)
        dates = self.date_pattern.findall(content)
        
        return {
            'content': content,
            'extracted_data': {
                'emails': emails,
                'phones': phones,
                'dates': dates,
                'word_count': len(content.split()),
                'char_count': len(content)
            }
        }

# Global extractor instance
data_extractor = DataExtractor()
