"""
Business Document Type Definitions

Defines the most common business document types and their characteristics
based on real-world business operations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

class BusinessDocumentType(Enum):
    """Most common business document types."""
    
    # Financial Documents
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    RECEIPT = "receipt"
    EXPENSE_REPORT = "expense_report"
    FINANCIAL_STATEMENT = "financial_statement"
    BUDGET_REPORT = "budget_report"
    
    # HR Documents
    RESUME = "resume"
    JOB_POSTING = "job_posting"
    EMPLOYEE_CONTRACT = "employee_contract"
    PERFORMANCE_REVIEW = "performance_review"
    TIMESHEET = "timesheet"
    LEAVE_REQUEST = "leave_request"
    
    # Communication
    BUSINESS_EMAIL = "business_email"
    MEMO = "memo"
    MEETING_MINUTES = "meeting_minutes"
    PROPOSAL = "proposal"
    REPORT = "report"
    PRESENTATION = "presentation"
    
    # Legal & Compliance
    CONTRACT = "contract"
    NDA = "nda"
    TERMS_OF_SERVICE = "terms_of_service"
    POLICY_DOCUMENT = "policy_document"
    COMPLIANCE_REPORT = "compliance_report"
    
    # Operations
    PROJECT_PLAN = "project_plan"
    WORK_ORDER = "work_order"
    INVENTORY_REPORT = "inventory_report"
    SHIPPING_DOCUMENT = "shipping_document"
    QUALITY_REPORT = "quality_report"
    
    # Sales & Marketing
    SALES_QUOTE = "sales_quote"
    MARKETING_BRIEF = "marketing_brief"
    CUSTOMER_FEEDBACK = "customer_feedback"
    PRODUCT_CATALOG = "product_catalog"
    PRICE_LIST = "price_list"

@dataclass
class DocumentPattern:
    """Pattern definition for document classification."""
    keywords: List[str]
    required_sections: List[str]
    common_phrases: List[str]
    regex_patterns: List[str]
    typical_length_range: Tuple[int, int]  # (min_words, max_words)
    confidence_threshold: float

# Business document patterns
BUSINESS_DOCUMENT_PATTERNS = {
    BusinessDocumentType.INVOICE: DocumentPattern(
        keywords=['invoice', 'bill to', 'ship to', 'total', 'tax', 'due date', 'payment terms'],
        required_sections=['bill_to', 'items', 'total'],
        common_phrases=['invoice number', 'due date', 'payment terms', 'subtotal', 'tax rate'],
        regex_patterns=[
            r'invoice\s*#?\s*:?\s*\w+',
            r'due\s+date\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'total\s*:?\s*\$?\s*[\d,]+\.?\d*'
        ],
        typical_length_range=(50, 500),
        confidence_threshold=0.85
    ),
    
    BusinessDocumentType.RESUME: DocumentPattern(
        keywords=['experience', 'education', 'skills', 'objective', 'references'],
        required_sections=['contact', 'experience', 'education'],
        common_phrases=['work experience', 'professional experience', 'education background', 'core skills'],
        regex_patterns=[
            r'\b\d{4}\s*-\s*(?:\d{4}|present)\b',  # Date ranges
            r'\b(?:bachelor|master|phd|degree)\b',
            r'\b\w+\s+university\b'
        ],
        typical_length_range=(200, 1500),
        confidence_threshold=0.80
    ),
    
    BusinessDocumentType.BUSINESS_EMAIL: DocumentPattern(
        keywords=['from', 'to', 'subject', 'sent', 'regards', 'sincerely'],
        required_sections=['from', 'to', 'subject'],
        common_phrases=['dear', 'thank you', 'please find attached', 'best regards', 'looking forward'],
        regex_patterns=[
            r'^from:\s*.+@.+\..+',
            r'^to:\s*.+@.+\..+',
            r'^subject:\s*.+'
        ],
        typical_length_range=(20, 500),
        confidence_threshold=0.90
    ),
    
    BusinessDocumentType.CONTRACT: DocumentPattern(
        keywords=['agreement', 'party', 'parties', 'terms', 'conditions', 'whereas', 'hereby'],
        required_sections=['parties', 'terms', 'signatures'],
        common_phrases=['this agreement', 'effective date', 'terms and conditions', 'governing law'],
        regex_patterns=[
            r'(?:first|second)\s+party',
            r'effective\s+(?:date|from)',
            r'witness\s+whereof'
        ],
        typical_length_range=(500, 5000),
        confidence_threshold=0.85
    ),
    
    BusinessDocumentType.MEETING_MINUTES: DocumentPattern(
        keywords=['meeting', 'attendees', 'agenda', 'action items', 'minutes', 'discussion'],
        required_sections=['attendees', 'agenda', 'discussion'],
        common_phrases=['meeting called to order', 'action items', 'next meeting', 'motion carried'],
        regex_patterns=[
            r'attendees?\s*:',
            r'action\s+items?\s*:',
            r'date\s*:\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        ],
        typical_length_range=(100, 1000),
        confidence_threshold=0.75
    ),
    
    BusinessDocumentType.PURCHASE_ORDER: DocumentPattern(
        keywords=['purchase order', 'po number', 'vendor', 'ship to', 'quantity', 'unit price'],
        required_sections=['vendor', 'items', 'total'],
        common_phrases=['purchase order number', 'delivery date', 'payment terms', 'item description'],
        regex_patterns=[
            r'p\.?o\.?\s*#?\s*:?\s*\w+',
            r'quantity\s*:?\s*\d+',
            r'unit\s+price\s*:?\s*\$?\s*[\d,]+\.?\d*'
        ],
        typical_length_range=(50, 400),
        confidence_threshold=0.85
    ),
    
    BusinessDocumentType.EXPENSE_REPORT: DocumentPattern(
        keywords=['expense', 'reimbursement', 'receipt', 'mileage', 'per diem', 'total expenses', 'subtotal', 'total due'],
        required_sections=['employee', 'expenses', 'total'],
        common_phrases=['expense report', 'business purpose', 'expense category', 'approval signature', 'expense details', 'receipts attached'],
        regex_patterns=[
            r'expense\s+report',
            r'employee\s+information',
            r'expense\s+details',
            r'amount\s*:?\s*\$?\s*[\d,]+\.?\d*',
            r'total\s+due\s*:?\s*\$?\s*[\d,]+\.?\d*',
            r'subtotal\s*:?\s*\$?\s*[\d,]+\.?\d*',
            r'mileage\s*:?\s*\d+\s*(?:miles?|km)?'
        ],
        typical_length_range=(50, 2500),
        confidence_threshold=0.60
    ),
    
    BusinessDocumentType.PROJECT_PLAN: DocumentPattern(
        keywords=['project', 'milestone', 'deliverable', 'timeline', 'resources', 'objectives'],
        required_sections=['objectives', 'timeline', 'deliverables'],
        common_phrases=['project scope', 'key milestones', 'resource allocation', 'risk assessment'],
        regex_patterns=[
            r'phase\s+\d+',
            r'milestone\s*:',
            r'deadline\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        ],
        typical_length_range=(200, 2000),
        confidence_threshold=0.75
    ),
    
    BusinessDocumentType.SALES_QUOTE: DocumentPattern(
        keywords=['quote', 'quotation', 'pricing', 'valid until', 'terms', 'conditions'],
        required_sections=['customer', 'items', 'total', 'validity'],
        common_phrases=['quotation number', 'valid until', 'payment terms', 'delivery terms'],
        regex_patterns=[
            r'quote\s*#?\s*:?\s*\w+',
            r'valid\s+(?:until|through)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'(?:sub)?total\s*:?\s*\$?\s*[\d,]+\.?\d*'
        ],
        typical_length_range=(100, 600),
        confidence_threshold=0.85
    ),
    
    BusinessDocumentType.TIMESHEET: DocumentPattern(
        keywords=['timesheet', 'hours', 'overtime', 'regular time', 'week ending', 'daily hours'],
        required_sections=['employee', 'period', 'hours'],
        common_phrases=['regular hours', 'overtime hours', 'total hours', 'supervisor approval'],
        regex_patterns=[
            r'week\s+ending\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?\s*:?\s*\d+\.?\d*',
            r'total\s+hours?\s*:?\s*\d+\.?\d*'
        ],
        typical_length_range=(30, 200),
        confidence_threshold=0.90
    )
}

def get_collection_name(doc_type: BusinessDocumentType) -> str:
    """Get the MongoDB collection name for a document type."""
    category_mapping = {
        # Financial
        BusinessDocumentType.INVOICE: "financial_invoices",
        BusinessDocumentType.PURCHASE_ORDER: "financial_orders",
        BusinessDocumentType.RECEIPT: "financial_receipts",
        BusinessDocumentType.EXPENSE_REPORT: "financial_expenses",
        BusinessDocumentType.FINANCIAL_STATEMENT: "financial_statements",
        BusinessDocumentType.BUDGET_REPORT: "financial_budgets",
        
        # HR
        BusinessDocumentType.RESUME: "hr_resumes",
        BusinessDocumentType.JOB_POSTING: "hr_job_postings",
        BusinessDocumentType.EMPLOYEE_CONTRACT: "hr_contracts",
        BusinessDocumentType.PERFORMANCE_REVIEW: "hr_reviews",
        BusinessDocumentType.TIMESHEET: "hr_timesheets",
        BusinessDocumentType.LEAVE_REQUEST: "hr_leave_requests",
        
        # Communication
        BusinessDocumentType.BUSINESS_EMAIL: "comm_emails",
        BusinessDocumentType.MEMO: "comm_memos",
        BusinessDocumentType.MEETING_MINUTES: "comm_meetings",
        BusinessDocumentType.PROPOSAL: "comm_proposals",
        BusinessDocumentType.REPORT: "comm_reports",
        BusinessDocumentType.PRESENTATION: "comm_presentations",
        
        # Legal
        BusinessDocumentType.CONTRACT: "legal_contracts",
        BusinessDocumentType.NDA: "legal_ndas",
        BusinessDocumentType.TERMS_OF_SERVICE: "legal_terms",
        BusinessDocumentType.POLICY_DOCUMENT: "legal_policies",
        BusinessDocumentType.COMPLIANCE_REPORT: "legal_compliance",
        
        # Operations
        BusinessDocumentType.PROJECT_PLAN: "ops_projects",
        BusinessDocumentType.WORK_ORDER: "ops_work_orders",
        BusinessDocumentType.INVENTORY_REPORT: "ops_inventory",
        BusinessDocumentType.SHIPPING_DOCUMENT: "ops_shipping",
        BusinessDocumentType.QUALITY_REPORT: "ops_quality",
        
        # Sales & Marketing
        BusinessDocumentType.SALES_QUOTE: "sales_quotes",
        BusinessDocumentType.MARKETING_BRIEF: "marketing_briefs",
        BusinessDocumentType.CUSTOMER_FEEDBACK: "sales_feedback",
        BusinessDocumentType.PRODUCT_CATALOG: "sales_catalogs",
        BusinessDocumentType.PRICE_LIST: "sales_pricing"
    }
    
    return category_mapping.get(doc_type, "business_general")

def get_document_category(doc_type: BusinessDocumentType) -> str:
    """Get the high-level category for a document type."""
    if doc_type.value in ['invoice', 'purchase_order', 'receipt', 'expense_report', 'financial_statement', 'budget_report']:
        return "Financial"
    elif doc_type.value in ['resume', 'job_posting', 'employee_contract', 'performance_review', 'timesheet', 'leave_request']:
        return "Human Resources"
    elif doc_type.value in ['business_email', 'memo', 'meeting_minutes', 'proposal', 'report', 'presentation']:
        return "Communication"
    elif doc_type.value in ['contract', 'nda', 'terms_of_service', 'policy_document', 'compliance_report']:
        return "Legal & Compliance"
    elif doc_type.value in ['project_plan', 'work_order', 'inventory_report', 'shipping_document', 'quality_report']:
        return "Operations"
    elif doc_type.value in ['sales_quote', 'marketing_brief', 'customer_feedback', 'product_catalog', 'price_list']:
        return "Sales & Marketing"
    else:
        return "General Business"