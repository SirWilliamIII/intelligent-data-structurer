#!/usr/bin/env python3
"""
Debug the similarity detection to see why clustering isn't working
"""

import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def debug_similarity():
    """Debug similarity calculation on current staging documents."""
    
    print("🔍 Debugging Similarity Detection")
    print("=" * 50)
    
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['intelligent_data']
    staging = db['document_staging']
    
    # Get all staged documents
    docs = list(staging.find())
    print(f"📊 Found {len(docs)} documents in staging")
    
    if len(docs) < 2:
        print("❌ Need at least 2 documents for similarity")
        return
    
    # Extract content
    contents = []
    filenames = []
    
    for doc in docs:
        content = doc.get('content', '')
        filename = doc.get('source_file', 'unknown')
        
        if len(content) > 50:  # Only substantial content
            contents.append(content)
            filenames.append(filename)
    
    print(f"📝 Processing {len(contents)} documents with substantial content:")
    for filename in filenames:
        print(f"   • {filename}")
    
    if len(contents) < 2:
        print("❌ Need at least 2 documents with substantial content")
        return
    
    # Calculate TF-IDF similarity
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Allow words that appear in only 1 document
            max_df=0.95  # Ignore words that appear in >95% of documents
        )
        
        tfidf_matrix = vectorizer.fit_transform(contents)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print(f"\n🔢 Similarity Matrix ({len(contents)}x{len(contents)}):")
        print("   ", end="")
        for i, filename in enumerate(filenames):
            print(f"{i:4d}", end="")
        print()
        
        for i, filename in enumerate(filenames):
            print(f"{i:2d}", end=" ")
            for j in range(len(filenames)):
                if i == j:
                    print(f"    ", end="")  # Skip diagonal
                else:
                    similarity = similarity_matrix[i][j]
                    print(f"{similarity:4.2f}", end="")
            print(f"  {filename[:30]}")
        
        # Find potential clusters
        threshold = 0.7
        print(f"\n🌱 Potential Clusters (similarity >= {threshold}):")
        
        clusters_found = False
        for i in range(len(filenames)):
            similar_docs = []
            for j in range(len(filenames)):
                if i != j and similarity_matrix[i][j] >= threshold:
                    similar_docs.append((j, similarity_matrix[i][j]))
            
            if similar_docs:
                clusters_found = True
                print(f"   📄 {filenames[i]} similar to:")
                for j, sim in similar_docs:
                    print(f"      • {filenames[j]} ({sim:.2f})")
        
        if not clusters_found:
            print(f"   ❌ No clusters found with threshold {threshold}")
            
            # Try lower threshold
            lower_threshold = 0.3
            print(f"\n🔍 Trying lower threshold ({lower_threshold}):")
            
            for i in range(len(filenames)):
                similar_docs = []
                for j in range(len(filenames)):
                    if i != j and similarity_matrix[i][j] >= lower_threshold:
                        similar_docs.append((j, similarity_matrix[i][j]))
                
                if similar_docs:
                    print(f"   📄 {filenames[i]} similar to:")
                    for j, sim in similar_docs:
                        print(f"      • {filenames[j]} ({sim:.2f})")
        
        # Show top features for understanding
        print(f"\n🏷️  Top TF-IDF Features:")
        feature_names = vectorizer.get_feature_names_out()
        for i, filename in enumerate(filenames[:3]):  # Show first 3 docs
            doc_tfidf = tfidf_matrix[i].toarray()[0]
            top_indices = doc_tfidf.argsort()[-10:][::-1]
            top_features = [feature_names[idx] for idx in top_indices if doc_tfidf[idx] > 0]
            print(f"   📄 {filename}: {', '.join(top_features[:5])}")
            
    except Exception as e:
        print(f"❌ TF-IDF calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_similarity()