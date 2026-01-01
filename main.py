"""
FastAPI main entry point for Parent Dashboard backend.
"""
import sys
from pathlib import Path

# Ensure the backend directory is in Python path for imports
# This makes the backend work regardless of where it's run from
BACKEND_DIR = Path(__file__).parent.absolute()
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from parentdashboard.api.routes import router as parent_router
from therapygeneration.api.routes import router as therapy_router

# Initialize FastAPI app
app = FastAPI(
    title="Parent Dashboard API",
    description="AI Assistant API for Parent Dashboard using RAG with Groq LLM",
    version="1.0.0"
)

# Configure CORS for Flutter frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(parent_router)
app.include_router(therapy_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Parent Dashboard API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404 errors in logs."""
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

