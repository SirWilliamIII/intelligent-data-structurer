#!/usr/bin/env python3
"""
Simple test script to test the new dynamic classification system.
"""
import asyncio
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.intelligent_analyzer import IntelligentAnalyzer

async def test_classification():
    """Test the classification system."""
    analyzer = IntelligentAnalyzer()
    
    # Test sports content
    sports_content = """Football
Basketball
Baseball
Soccer
Tennis
Swimming
Golf
Hockey
Boxing
Wrestling
Track and Field
Volleyball
Cricket
Rugby
American Football
Table Tennis
Badminton
Cycling
Gymnastics
Skiing"""
    
    print("Testing sports classification...")
    result = await analyzer.analyze_content(sports_content, "sports.txt")
    
    print(f"Collection Name: {result.table_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Domain Keywords: {result.extracted_data.get('semantic_signature', {}).get('domain_keywords', [])}")
    print(f"Structural Patterns: {result.extracted_data.get('semantic_signature', {}).get('structural_patterns', [])}")
    print(f"Content Type: {result.extracted_data.get('content_type', 'unknown')}")
    print("\n" + "="*50 + "\n")
    
    # Test technical content
    tech_content = """kubectl get pods
kubectl apply -f deployment.yaml
kubectl describe service my-service
kubectl logs pod-name
kubectl exec -it pod-name -- /bin/bash
kubectl delete deployment my-deployment
kubectl scale deployment my-deployment --replicas=3"""
    
    print("Testing technical content classification...")
    result = await analyzer.analyze_content(tech_content, "kubernetes-cheat-sheet.txt")
    
    print(f"Collection Name: {result.table_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Domain Keywords: {result.extracted_data.get('semantic_signature', {}).get('domain_keywords', [])}")
    print(f"Structural Patterns: {result.extracted_data.get('semantic_signature', {}).get('structural_patterns', [])}")
    print(f"Content Type: {result.extracted_data.get('content_type', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(test_classification())