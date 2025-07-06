#!/usr/bin/env python3
"""
Main application entry point for the Intelligent Data Processor.
"""

import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from core.config import settings
from core.intelligent_analyzer import IntelligentAnalyzer
from core.database import DatabaseManager
from core.document_processor import DocumentProcessor

app = FastAPI(
    title="Intelligent Data Processor",
    description="Automatically classify and structure unstructured data",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize intelligent analyzer, database manager, and document processor
intelligent_analyzer = IntelligentAnalyzer()
db_manager = DatabaseManager()
document_processor = DocumentProcessor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Modern home page with beautiful UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify")
async def classify_content(file: UploadFile = File(...)):
    """Classify uploaded content."""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Intelligent analysis (without storing)
        result = await intelligent_analyzer.analyze_content(text_content, file.filename)
        
        return {
            "filename": file.filename,
            "table_name": result.table_name,
            "domain": result.schema_evolution.table_name.split('_')[0] if '_' in result.schema_evolution.table_name else 'general',
            "content_type": result.extracted_data.get('content_type', 'document'),
            "confidence": result.confidence,
            "priority": result.extracted_data.get('priority', 5.0),
            "reasoning": result.reasoning,
            "similar_content_found": len(result.similar_content_ids),
            "learning_feedback": result.learning_feedback,
            "content_markers": result.extracted_data.get('content_markers', []),
            "structural_patterns": result.extracted_data.get('structural_patterns', []),
            "suggested_schema": result.schema_evolution.suggested_additions,
            "entities_found": len(result.extracted_data.get('entities', [])),
            "status": "analyzed_only"
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text-based")
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process")
async def process_and_store_content(file: UploadFile = File(...)):
    """Intelligently analyze, classify, and store uploaded content with learning."""
    logger.info(f"Processing file: {file.filename}, size: {file.size}, content_type: {file.content_type}")
    try:
        # Read file content
        content = await file.read()
        
        # Extract text using document processor (handles PDFs, images, text)
        text_content, extraction_metadata = await document_processor.extract_text(
            content, file.filename, file.content_type
        )
        
        # Intelligent analysis with learning
        result = await intelligent_analyzer.analyze_content(text_content, file.filename)
        
        # Create table with intelligent schema evolution
        table_name = await db_manager.create_intelligent_table(
            result.table_name, 
            result.schema_evolution
        )
        
        # Store structured data
        await db_manager.insert_structured_data(table_name, result.extracted_data)
        
        return {
            "filename": file.filename,
            "table_name": result.table_name,
            "domain": result.schema_evolution.table_name.split('_')[0] if '_' in result.schema_evolution.table_name else 'general',
            "content_type": result.extracted_data.get('content_type', 'document'),
            "confidence": result.confidence,
            "priority": result.extracted_data.get('priority', 5.0),
            "reasoning": result.reasoning,
            "similar_content_found": len(result.similar_content_ids),
            "learning_feedback": result.learning_feedback,
            "schema_suggestions": result.schema_evolution.suggested_additions,
            "extraction_metadata": extraction_metadata,  # Include file processing details
            "extracted_data": {
                k: v for k, v in result.extracted_data.items() 
                if k not in ['content', 'semantic_signature']  # Exclude large data from response
            },
            "status": "stored_and_learned"
        }
        
    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error for file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail="Could not extract text from file")
    except ImportError as e:
        logger.error(f"Missing dependencies for file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Missing dependencies: {str(e)}")
    except Exception as e:
        logger.error(f"Processing error for file {file.filename}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/database/tables")
async def list_database_tables():
    """List all created database tables."""
    try:
        tables = await db_manager.list_tables()
        stats = await db_manager.get_table_stats()
        
        return {
            "tables": [
                {
                    "name": table,
                    "row_count": stats.get(table, 0)
                }
                for table in tables
            ],
            "total_tables": len(tables),
            "total_rows": sum(stats.values())
        }
    except Exception as e:
        logger.error(f"Failed to get tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/database/tables/{table_name}")
async def get_table_data(table_name: str, limit: int = 50):
    """Get data from a specific table."""
    try:
        data = await db_manager.get_table_data(table_name, limit)
        return {
            "table_name": table_name,
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        logger.error(f"Failed to get table data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "classifier_loaded": classifier.nlp is not None,
        "database_connected": True,
        "settings": {
            "confidence_threshold": settings.confidence_threshold,
            "spacy_model": settings.spacy_model
        }
    }

if __name__ == "__main__":
    logger.info("Starting Intelligent Data Processor")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
