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
from core.intelligent_collection_engine import IntelligentCollectionEngine
from core.document_processor import DocumentProcessor

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
mongo_client = None
intelligent_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB and intelligent system on startup."""
    global mongo_client, intelligent_engine
    
    try:
        mongo_client = pymongo.MongoClient(settings.mongo_url)
        intelligent_engine = IntelligentCollectionEngine(mongo_client)
        logger.info("🧠 Intelligent Collection Engine initialized")
        logger.info("🎯 Priority: Intelligence and Accuracy over Speed")
    except Exception as e:
        logger.error(f"Failed to initialize intelligent system: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with intelligent system info."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_with_intelligent_system(file: UploadFile = File(...)):
    """Process document through the INTELLIGENT collection system."""
    logger.info(f"🧠 Processing file with INTELLIGENT system: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text using document processor
        text_content, extraction_metadata = await document_processor.extract_text(
            content, file.filename, file.content_type
        )
        
        # Create document data for intelligent processing
        document_data = {
            "source_file": file.filename,
            "content": text_content,
            "inserted_at": None  # Will be set by processor
        }
        
        # Process through INTELLIGENT system
        result = await intelligent_engine.process_document(document_data)
        
        # Get system statistics
        stats = await intelligent_engine.get_processing_statistics()
        
        return {
            "filename": file.filename,
            "collection_assigned": result.collection_assigned,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "reasoning": result.reasoning,
            "extraction_metadata": extraction_metadata,
            
            # Intelligent system insights
            "claude_analysis": {
                "entities_found": len(result.ontology_analysis.entities) if result.ontology_analysis else 0,
                "document_type": result.ontology_analysis.document_type.type_name if result.ontology_analysis and result.ontology_analysis.document_type else "unknown",
                "domain": result.ontology_analysis.document_type.domain if result.ontology_analysis and result.ontology_analysis.document_type else "unknown",
                "concept_hierarchies": len(result.ontology_analysis.concept_hierarchies) if result.ontology_analysis else 0
            },
            "drift_detected": result.drift_signal is not None,
            "new_collections_created": result.created_collections,
            
            # Processing statistics
            "system_stats": {
                "total_documents": stats["processing_stats"]["total_documents"],
                "average_confidence": stats["processing_stats"]["average_confidence"],
                "collections_created": stats["processing_stats"]["new_collections_created"]
            },
            
            "status": "processed_intelligently"
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
        
        # Analyze with intelligent system (Claude ontology analysis)
        ontology_analysis = await intelligent_engine.claude_analyzer.analyze_document(
            text_content, file.filename
        )
        
        # Get placement decision
        placement = await intelligent_engine.taxonomy_manager.evaluate_document_placement(
            ontology_analysis
        )
        
        return {
            "filename": file.filename,
            "predicted_collection": placement.target_collection,
            "confidence": placement.confidence,
            "reasoning": placement.reasoning,
            "document_type": ontology_analysis.document_type.type_name if ontology_analysis.document_type else "unknown",
            "domain": ontology_analysis.document_type.domain if ontology_analysis.document_type else "unknown",
            "entities_found": len(ontology_analysis.entities),
            "concept_hierarchies": len(ontology_analysis.concept_hierarchies),
            "claude_confidence": ontology_analysis.confidence_score,
            "status": "classified_intelligently"
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/ecosystem")
async def get_ecosystem_status():
    """Get current intelligent ecosystem status."""
    try:
        stats = await intelligent_engine.get_processing_statistics()
        taxonomy_overview = await intelligent_engine.taxonomy_manager.get_taxonomy_overview()
        
        return {
            "ecosystem": {
                "processing_stats": stats["processing_stats"],
                "taxonomy_structure": taxonomy_overview,
                "system_health": stats["system_health"]
            },
            "system_type": "intelligent_taxonomy",
            "claude_powered": True,
            "concept_drift_detection": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get ecosystem status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all collections in the intelligent ecosystem."""
    try:
        # Get collections from collection registry
        registry = intelligent_engine.taxonomy_manager.collection_registry
        collections = list(registry.find({"status": "active"}))
        
        # Get detailed collection info
        collections_info = []
        total_documents = 0
        
        for collection in collections:
            collection_name = collection["collection_name"]
            doc_count = collection.get("document_count", 0)
            total_documents += doc_count
            
            collection_info = {
                "name": collection_name,
                "document_count": doc_count,
                "type": "intelligent",
                "abstraction_level": collection.get("hierarchy", {}).get("abstraction_level", "medium"),
                "parent": collection.get("hierarchy", {}).get("parent_collection"),
                "children": collection.get("hierarchy", {}).get("child_collections", []),
                "confidence": collection.get("confidence_score", 0.0),
                "created_at": collection.get("created_at", "unknown")
            }
            
            # Add ontology profile info
            ontology_profile = collection.get("ontology_profile", {})
            if ontology_profile:
                collection_info["domains"] = ontology_profile.get("domains", [])
                collection_info["document_types"] = ontology_profile.get("document_types", [])
                collection_info["primary_concepts"] = ontology_profile.get("primary_concepts", [])
            
            collections_info.append(collection_info)
        
        # Add staging info
        staging_count = intelligent_engine.db.document_staging.count_documents({})
        if staging_count > 0:
            collections_info.append({
                "name": "document_staging",
                "document_count": staging_count,
                "type": "staging",
                "description": "Documents awaiting intelligent processing"
            })
            total_documents += staging_count
        
        return {
            "collections": collections_info,
            "total_collections": len(collections),
            "total_documents": total_documents,
            "system_type": "intelligent_taxonomy",
            "claude_powered": True
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
async def trigger_intelligent_analysis():
    """Manually trigger intelligent taxonomy analysis."""
    try:
        logger.info("🧠 Manually triggering intelligent taxonomy analysis...")
        
        # Force baseline updates for drift detection
        await intelligent_engine.drift_detector.update_baselines(force_update=True)
        
        # Get system statistics
        stats = await intelligent_engine.get_processing_statistics()
        
        # Validate system health
        health = await intelligent_engine.validate_system_health()
        
        return {
            "analysis_triggered": True,
            "processing_stats": stats["processing_stats"],
            "taxonomy_overview": stats["taxonomy_overview"],
            "system_health": health,
            "claude_powered": True
        }
        
    except Exception as e:
        logger.error(f"Intelligent analysis trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ecosystem/insights")
async def ecosystem_insights():
    """Get insights about the organic collection ecosystem."""
    try:
        return await intelligent_engine.get_ecosystem_insights()
    except Exception as e:
        logger.error(f"Failed to get ecosystem insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check for intelligent system."""
    
    intelligent_healthy = intelligent_engine is not None
    mongo_healthy = mongo_client is not None
    
    try:
        if intelligent_healthy and mongo_healthy:
            health_report = await intelligent_engine.validate_system_health()
            ecosystem_health = 100 if health_report["overall_status"] == "healthy" else 50
        else:
            ecosystem_health = 0
    except:
        ecosystem_health = 0
    
    return {
        "status": "healthy" if intelligent_healthy and mongo_healthy else "degraded",
        "intelligent_system": intelligent_healthy,
        "mongodb_connected": mongo_healthy,
        "ecosystem_health_percentage": ecosystem_health,
        "version": "3.0.0 - Intelligent Taxonomy",
        "features": [
            "claude_powered_analysis",
            "concept_drift_detection", 
            "dynamic_taxonomy_management",
            "semantic_understanding",
            "hierarchical_organization",
            "volume_based_triggers"
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