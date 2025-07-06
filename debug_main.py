#!/usr/bin/env python3
import uvicorn
from loguru import logger

print("Script started")
logger.info("Logger works")
print("About to run uvicorn...")

try:
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=8002,
        reload=False
    )
    print("uvicorn.run returned")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()