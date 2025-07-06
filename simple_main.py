#!/usr/bin/env python3
"""
Simple version of the data processor without AI dependencies.
"""

import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from core.database import DatabaseManager

app = FastAPI(
    title="Intelligent Data Processor",
    description="Database management interface",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database manager
db_manager = DatabaseManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})

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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database_connected": True
    }

if __name__ == "__main__":
    uvicorn.run(
        "simple_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )