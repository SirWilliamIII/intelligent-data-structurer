#!/usr/bin/env python3
"""
Test the content cleaning functions to see what's actually happening.
"""

import sys
sys.path.append('.')

from core.intelligent_analyzer import IntelligentAnalyzer
from core.document_processor import DocumentProcessor

def test_cleaning():
    analyzer = IntelligentAnalyzer()
    processor = DocumentProcessor()
    
    # Test with some sample problematic content
    test_content = """


    
    
Hello world


    This is a test



    
    
With lots of whitespace


    
    


    """
    
    print("ORIGINAL CONTENT:")
    print(repr(test_content))
    print("\nORIGINAL LENGTH:", len(test_content))
    
    cleaned = analyzer._clean_content(test_content)
    
    print("\nCLEANED CONTENT:")
    print(repr(cleaned))
    print("\nCLEANED LENGTH:", len(cleaned))
    
    print("\nCLEANED READABLE:")
    print(cleaned)

if __name__ == "__main__":
    test_cleaning()