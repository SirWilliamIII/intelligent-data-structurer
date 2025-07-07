#!/usr/bin/env python3
"""
Quick fix to improve technical document classification
"""

import re
from typing import Set

def enhanced_technical_detection(content: str, filename: str) -> tuple[Set[str], Set[str], float]:
    """
    Enhanced detection for technical documents.
    Returns: (content_markers, structural_patterns, confidence)
    """
    
    content_lower = content.lower()
    filename_lower = filename.lower()
    
    content_markers = set()
    structural_patterns = set()
    confidence = 0.0
    
    # ENHANCED TECHNICAL MARKERS
    technical_keywords = [
        # Programming languages
        'python', 'javascript', 'java', 'cpp', 'csharp', 'go', 'rust', 'ruby', 'php',
        'typescript', 'scala', 'kotlin', 'swift', 'dart', 'lua', 'perl', 'bash',
        
        # Technologies & Frameworks
        'react', 'vue', 'angular', 'django', 'flask', 'express', 'nodejs', 'npm',
        'webpack', 'babel', 'jest', 'cypress', 'selenium', 'junit', 'maven', 'gradle',
        
        # DevOps & Cloud
        'aws', 'azure', 'gcp', 'terraform', 'ansible', 'jenkins', 'cicd', 'github',
        'gitlab', 'bitbucket', 'jira', 'confluence', 'slack', 'teams',
        
        # Database & Storage
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite',
        'cassandra', 'dynamodb', 'firebase', 'supabase',
        
        # Infrastructure
        'linux', 'ubuntu', 'centos', 'debian', 'windows', 'macos', 'nginx', 'apache',
        'microservices', 'serverless', 'containerization', 'orchestration',
        
        # Development concepts
        'api', 'rest', 'graphql', 'websocket', 'oauth', 'jwt', 'cors', 'csrf',
        'https', 'ssl', 'tls', 'encryption', 'authentication', 'authorization',
        
        # File formats & protocols
        'json', 'xml', 'yaml', 'csv', 'http', 'https', 'ftp', 'ssh', 'tcp', 'udp',
        
        # Commands & Tools
        'command', 'terminal', 'shell', 'script', 'config', 'configuration',
        'install', 'setup', 'deployment', 'build', 'compile', 'debug', 'test',
        'version', 'update', 'upgrade', 'patch', 'release', 'branch', 'merge',
        
        # Technical documentation terms
        'documentation', 'tutorial', 'guide', 'reference', 'manual', 'cheatsheet',
        'quickstart', 'getting-started', 'howto', 'faq', 'troubleshooting',
        'best-practices', 'guidelines', 'standards', 'conventions',
        
        # Development lifecycle
        'development', 'testing', 'staging', 'production', 'environment',
        'variable', 'parameter', 'argument', 'function', 'method', 'class',
        'object', 'interface', 'implementation', 'abstraction', 'inheritance',
        
        # Security
        'security', 'vulnerability', 'exploit', 'firewall', 'penetration',
        'audit', 'compliance', 'gdpr', 'hipaa', 'iso27001',
        
        # Monitoring & Logging
        'monitoring', 'logging', 'metrics', 'alerts', 'dashboard', 'grafana',
        'prometheus', 'kibana', 'splunk', 'datadog', 'newrelic'
    ]
    
    # Check for technical keywords
    tech_keyword_matches = 0
    for keyword in technical_keywords:
        if keyword in content_lower or keyword in filename_lower:
            tech_keyword_matches += 1
            content_markers.add('technical_content')
    
    # ENHANCED STRUCTURAL PATTERNS
    
    # Code blocks (various formats)
    if (re.search(r'```[\w]*\n.*?\n```', content, re.DOTALL) or
        re.search(r'`[^`\n]+`', content) or
        re.search(r'^\s{4,}[^\s]', content, re.MULTILINE)):
        structural_patterns.add('code_blocks')
        confidence += 0.3
    
    # Command line patterns
    if (re.search(r'^\s*\$\s+\w+', content, re.MULTILINE) or
        re.search(r'^\s*#\s+\w+', content, re.MULTILINE) or
        re.search(r'sudo\s+\w+', content) or
        re.search(r'npm\s+\w+', content) or
        re.search(r'pip\s+install', content)):
        structural_patterns.add('command_line')
        confidence += 0.3
    
    # Configuration file patterns
    if (re.search(r'^\w+\s*[:=]\s*\w+', content, re.MULTILINE) or
        re.search(r'^\s*\w+\s*\{', content, re.MULTILINE) or
        re.search(r'^\s*\[[\w\s]+\]', content, re.MULTILINE)):
        structural_patterns.add('configuration')
        confidence += 0.2
    
    # API documentation patterns
    if (re.search(r'(GET|POST|PUT|DELETE|PATCH)\s+/', content) or
        re.search(r'endpoint|api|request|response', content, re.IGNORECASE) or
        re.search(r'status\s+code|http\s+\d{3}', content, re.IGNORECASE)):
        structural_patterns.add('api_documentation')
        confidence += 0.3
    
    # Technical headers (markdown)
    if (re.search(r'^#{1,6}\s+(?:Installation|Configuration|Usage|API|Setup|Getting Started|Quick Start)', content, re.MULTILINE | re.IGNORECASE) or
        re.search(r'^#{1,6}\s+(?:Requirements|Dependencies|Prerequisites|Examples|Tutorial)', content, re.MULTILINE | re.IGNORECASE)):
        structural_patterns.add('technical_headers')
        confidence += 0.2
    
    # File paths and URLs
    if (re.search(r'/[a-zA-Z0-9_/-]+\.[a-zA-Z0-9]+', content) or
        re.search(r'https?://[^\s]+', content) or
        re.search(r'[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+', content)):
        structural_patterns.add('file_references')
        confidence += 0.1
    
    # Technical filename patterns
    tech_filename_patterns = [
        r'\.md$', r'\.txt$', r'\.py$', r'\.js$', r'\.json$', r'\.yaml$', r'\.yml$',
        r'readme', r'guide', r'tutorial', r'cheat', r'reference', r'manual',
        r'config', r'setup', r'install', r'deploy', r'build', r'test'
    ]
    
    for pattern in tech_filename_patterns:
        if re.search(pattern, filename_lower):
            structural_patterns.add('technical_filename')
            confidence += 0.1
            break
    
    # Boost confidence based on technical keyword density
    if tech_keyword_matches > 0:
        confidence += min(tech_keyword_matches * 0.05, 0.4)
    
    # Special boost for obvious technical files
    if tech_keyword_matches > 5:
        content_markers.add('highly_technical')
        confidence += 0.3
    
    return content_markers, structural_patterns, confidence

def test_classification():
    """Test the enhanced classification on sample tech content"""
    
    # Test cases
    test_cases = [
        ("kubernetes-guide.md", "# Kubernetes Guide\n\n## Installation\n\n```bash\nkubectl apply -f deployment.yaml\n```\n\nThis guide covers Docker containers and microservices."),
        ("react-tutorial.md", "# React Tutorial\n\n## Setup\n\n```javascript\nnpm install react\n```\n\nLearn about components, hooks, and state management."),
        ("sports-list.txt", "Football Teams:\n\nDallas Cowboys\nNew York Giants\nGreen Bay Packers\nChicago Bears"),
        ("api-reference.md", "# API Reference\n\n## Endpoints\n\nGET /api/users\nPOST /api/users\n\nStatus codes: 200, 404, 500")
    ]
    
    for filename, content in test_cases:
        markers, patterns, confidence = enhanced_technical_detection(content, filename)
        
        print(f"\n📁 {filename}")
        print(f"   🏷️  Content markers: {markers}")
        print(f"   📊 Structural patterns: {patterns}")
        print(f"   🎯 Confidence: {confidence:.1%}")
        
        # Determine if it should be technical
        is_technical = confidence > 0.4 or 'technical_content' in markers
        suggested_collection = 'technical_documents' if is_technical else 'general_documents'
        print(f"   📂 Suggested collection: {suggested_collection}")

if __name__ == "__main__":
    test_classification()