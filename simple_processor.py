"""
Simple, working document processor that actually cleans text properly.
"""

import re
import json
from typing import Dict, Any

def clean_text_properly(text: str) -> str:
    """Actually clean text - no bullshit."""
    
    if not text:
        return ""
    
    # Remove all control characters and null bytes
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    
    # Split into words, filter out garbage
    words = text.split()
    clean_words = []
    
    for word in words:
        # Must have at least one letter
        if re.search(r'[a-zA-Z]', word):
            # Remove excessive punctuation
            word = re.sub(r'[^\w\s\-\.\,\!\?]', '', word)
            if len(word) > 1:  # Skip single characters
                clean_words.append(word)
    
    # Join with single spaces, limit length
    result = ' '.join(clean_words)
    
    # If too long, truncate
    if len(result) > 2000:
        result = result[:2000] + "... [TRUNCATED]"
    
    return result

def extract_simple_data(text: str, filename: str) -> Dict[str, Any]:
    """Extract basic data without overthinking."""
    
    clean_content = clean_text_properly(text)
    
    return {
        'content': clean_content,
        'filename': filename,
        'word_count': len(clean_content.split()),
        'char_count': len(clean_content),
        'has_emails': '@' in clean_content,
        'has_urls': 'http' in clean_content.lower(),
        'has_dates': bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', clean_content))
    }

if __name__ == "__main__":
    # Test with garbage text
    garbage = "Hello\x00\x01world\n\n\n\n    lots    of    spaces\t\t\tand\x7f\x80weird\ufffdchars"
    print("Original:", repr(garbage))
    print("Cleaned:", repr(clean_text_properly(garbage)))