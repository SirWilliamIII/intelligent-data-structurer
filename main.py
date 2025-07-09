#!/usr/bin/env python3
"""
Organic Main Application - Uses the brilliant organic collection system
"""

import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn
import pymongo

from core.config import settings
from core.organic_collections import OrganicCollectionManager
from core.intelligent_analyzer import IntelligentAnalyzer
from core.document_processor import DocumentProcessor
from core.business_classifier import BusinessDocumentClassifier

app = FastAPI(
    title="Organic Intelligent Data Processor",
    description="Self-organizing document intelligence with organic collections",
    version="2.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
document_processor = DocumentProcessor()
intelligent_analyzer = IntelligentAnalyzer()
business_classifier = BusinessDocumentClassifier()
mongo_client = None
organic_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB and organic system on startup."""
    global mongo_client, organic_manager
    
    try:
        mongo_client = pymongo.MongoClient(settings.mongo_url)
        organic_manager = OrganicCollectionManager(mongo_client)
        logger.info("🌱 Organic collection system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize organic system: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with organic system info."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_with_organic_system(file: UploadFile = File(...)):
    """Process document through the organic collection system."""
    logger.info(f"🔄 Processing file with organic system: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text using document processor
        text_content, extraction_metadata = await document_processor.extract_text(
            content, file.filename, file.content_type
        )
        
        # Analyze with intelligent analyzer
        result = await intelligent_analyzer.analyze_content(text_content, file.filename)
        
        # Process through organic system
        organic_doc_id = await organic_manager.process_document(result.extracted_data)
        
        # Get ecosystem insights
        insights = await organic_manager.get_collection_insights()
        
        return {
            "filename": file.filename,
            "organic_doc_id": organic_doc_id,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "extraction_metadata": extraction_metadata,
            
            # Organic system insights
            "ecosystem_status": insights["ecosystem_status"],
            "collection_health": insights["health_summary"],
            
            # What happened during processing
            "processing_result": {
                "staged": True,
                "seeds_evaluated": True,
                "similar_docs_found": len(result.similar_content_ids),
                "learning_feedback": result.learning_feedback
            },
            
            "status": "processed_organically"
        }
        
    except Exception as e:
        logger.error(f"Organic processing error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/classify")
async def classify_only(file: UploadFile = File(...)):
    """Classify without storing (for preview)."""
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Analyze with intelligent analyzer
        result = await intelligent_analyzer.analyze_content(text_content, file.filename)
        
        return {
            "filename": file.filename,
            "predicted_collection": result.table_name,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "content_markers": result.extracted_data.get('content_markers', []),
            "structural_patterns": result.extracted_data.get('structural_patterns', []),
            "entities_found": len(result.extracted_data.get('entities', [])),
            "word_count": result.extracted_data.get('word_count', 0),
            "status": "classified_only"
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/ecosystem")
async def get_ecosystem_status():
    """Get current organic ecosystem status."""
    try:
        insights = await organic_manager.get_collection_insights()
        suggestions = await organic_manager.suggest_collection_improvements()
        
        return {
            "ecosystem": insights,
            "suggestions": suggestions,
            "system_type": "organic_evolution"
        }
        
    except Exception as e:
        logger.error(f"Failed to get ecosystem status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all collections in the ecosystem."""
    try:
        insights = await organic_manager.get_collection_insights()
        
        # Get detailed collection info
        collections_info = []
        
        for collection_name, doc_count in insights["collection_sizes"].items():
            collection_info = {
                "name": collection_name,
                "document_count": doc_count,
                "type": "mature"
            }
            
            # Add health info if available
            if collection_name in organic_manager.collection_health:
                health = organic_manager.collection_health[collection_name]
                collection_info["health"] = {
                    "coherence_score": health.coherence_score,
                    "growth_rate": health.growth_rate,
                    "diversity_index": health.diversity_index
                }
            
            # Add DNA info if available  
            if collection_name in organic_manager.collection_dna:
                dna = organic_manager.collection_dna[collection_name]
                collection_info["themes"] = list(dna.core_themes)[:5]
                collection_info["patterns"] = list(dna.structural_patterns)
            
            collections_info.append(collection_info)
        
        # Add staging info
        staging_count = insights["ecosystem_status"]["documents_in_staging"]
        if staging_count > 0:
            collections_info.append({
                "name": "document_staging",
                "document_count": staging_count,
                "type": "staging",
                "description": "Documents waiting for classification or clustering"
            })
        
        # Add seeds info
        seed_count = insights["ecosystem_status"]["collection_seeds"]
        if seed_count > 0:
            collections_info.append({
                "name": "collection_seeds",
                "document_count": seed_count,
                "type": "seeds",
                "description": "Potential collections forming from similar documents"
            })
        
        return {
            "collections": collections_info,
            "total_mature": insights["ecosystem_status"]["mature_collections"],
            "total_documents": insights["ecosystem_status"]["total_organized_documents"] + staging_count,
            "ecosystem_health": insights["health_summary"]["health_percentage"]
        }
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{collection_name}")
async def get_collection_data(collection_name: str, limit: int = 50):
    """Get documents from a specific collection."""
    try:
        db = mongo_client[settings.database_name]
        collection = db[collection_name]
        
        # Get documents
        docs = list(collection.find().limit(limit))
        
        # Convert ObjectIds to strings for JSON serialization
        for doc in docs:
            doc['_id'] = str(doc['_id'])
        
        return {
            "collection_name": collection_name,
            "documents": docs,
            "count": len(docs),
            "total_count": collection.count_documents({})
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evolve")
async def trigger_evolution():
    """Manually trigger evolution process."""
    try:
        logger.info("🧬 Manually triggering evolution...")
        
        # Force evolution cycles
        await organic_manager._evolve_collections()
        await organic_manager._health_check_collections()
        
        # Get updated insights
        insights = await organic_manager.get_collection_insights()
        suggestions = await organic_manager.suggest_collection_improvements()
        
        return {
            "evolution_triggered": True,
            "ecosystem_after": insights,
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Evolution trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check for organic system."""
    
    organic_healthy = organic_manager is not None
    mongo_healthy = mongo_client is not None
    
    try:
        if organic_healthy and mongo_healthy:
            insights = await organic_manager.get_collection_insights()
            ecosystem_health = insights["health_summary"]["health_percentage"]
        else:
            ecosystem_health = 0
    except:
        ecosystem_health = 0
    
    return {
        "status": "healthy" if organic_healthy and mongo_healthy else "degraded",
        "organic_system": organic_healthy,
        "mongodb_connected": mongo_healthy,
        "ecosystem_health_percentage": ecosystem_health,
        "version": "2.0.0 - Organic Evolution",
        "features": [
            "self_organizing_collections",
            "intelligent_clustering", 
            "adaptive_schemas",
            "health_monitoring",
            "similarity_learning"
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Organic Intelligent Data Processor v2.0")
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8000,  # Main port
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )