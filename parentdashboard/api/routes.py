"""
API routes for Parent Dashboard.
"""
import logging
import time
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional

from parentdashboard.config import PDFS_DIR
from parentdashboard.schemas.request import QuestionRequest, UpdatePdfRequest
from parentdashboard.schemas.response import AnswerResponse
from parentdashboard.schemas.speech_stats import (
    SpeechStatsResponse,
    ChildSummaryResponse,
)
from parentdashboard.services.qa_service import QAService
from parentdashboard.services.service import (
    FirestoreSpeechRepository,
    SpeechStatsService,
    get_dashboard_stats_cached,
    get_latest_activity_timestamp,
    _resolve_child_uid,
    LOGGED_IN_USER_UID,
    get_accuracy_from_latest_practice_per_session,
    get_monthly_practice_count,
    get_target_sounds_last_4_sessions,
)
logger = logging.getLogger(__name__)

# In-memory cache for /child-summary: child_id -> (response_dict, expiry_timestamp, last_activity_ts)
# If DB is updated within TTL we invalidate so the next request uses updated data.
_child_summary_response_cache: Dict[str, Tuple[Dict[str, Any], float, Any]] = {}
_CHILD_SUMMARY_RESPONSE_TTL = 90  # seconds

# Initialize router
router = APIRouter(prefix="/parentdashboard", tags=["Parent Dashboard"])

# Initialize services (singleton-style instances)
qa_service = QAService()
_speech_repository = FirestoreSpeechRepository()
_speech_stats_service = SpeechStatsService(repository=_speech_repository)


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the AI assistant.
    
    Args:
        request: QuestionRequest containing the parent's question
    
    Returns:
        AnswerResponse with the AI-generated answer
    """
    try:
        child_id = (request.child_id or LOGGED_IN_USER_UID).strip() or LOGGED_IN_USER_UID
        result = qa_service.answer_question(
            request.question,
            child_id=child_id,
        )
        return AnswerResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.get("/speech-progress", response_model=SpeechStatsResponse)
async def get_speech_progress(child_id: Optional[str] = None):
    """
    Get speech accuracy statistics for a child.

    Currently uses a mock repository that simulates Firestore documents.
    The repository is injected into the service so it can be swapped with
    a real Firestore-backed implementation later without changing this
    route or the frontend integration.
    """
    try:
        return _speech_stats_service.get_stats(child_id=child_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating speech progress: {str(e)}",
        )


@router.get("/child-summary", response_model=ChildSummaryResponse)
async def get_child_summary(child_id: Optional[str] = None):
    """
    Get high-level child information and progress for the parent dashboard.
    Response is cached in memory for 90s to speed up repeat loads.
    """
    try:
        child_id_value = child_id or LOGGED_IN_USER_UID
        now = time.time()
        if child_id_value in _child_summary_response_cache:
            cached, expiry, last_ts = _child_summary_response_cache[child_id_value]
            if now < expiry:
                current_ts = get_latest_activity_timestamp(child_id_value)
                if current_ts == last_ts:
                    return ChildSummaryResponse(**cached)
            del _child_summary_response_cache[child_id_value]

        # Child metadata from users/{resolved_uid}; default to logged-in user when child_id not sent
        from parentdashboard.data.firebase_client import get_firestore_client

        client = get_firestore_client()
        resolved_uid = _resolve_child_uid(child_id_value)
        doc_ref = client.collection("users").document(resolved_uid or child_id_value)
        doc = doc_ref.get()
        raw = doc.to_dict() or {} if doc.exists else {}

        name = raw.get("name", "Unknown Child")
        age = int(raw.get("age", 0) or 0)

        overall_accuracy = get_accuracy_from_latest_practice_per_session(child_id_value)
        monthly_practice_count = get_monthly_practice_count(child_id_value)
        target_sounds = get_target_sounds_last_4_sessions(child_id_value)

        response_data = {
            "id": child_id_value,
            "name": name,
            "age": age,
            "overall_accuracy": round(overall_accuracy, 2),
            "monthly_practice_count": monthly_practice_count,
            "target_sounds": target_sounds,
        }
        last_ts = get_latest_activity_timestamp(child_id_value)
        _child_summary_response_cache[child_id_value] = (
            response_data,
            now + _CHILD_SUMMARY_RESPONSE_TTL,
            last_ts,
        )
        return ChildSummaryResponse(**response_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating child summary: {str(e)}",
        )


@router.get("/child-stats/{child_id}")
async def get_child_stats(child_id: str, user_id: Optional[str] = None):
    """
    Get high-level dashboard statistics for a specific child.

    child_id is the child's Firebase UID (users/{child_id}/sessions/...).
    user_id is optional and not used for the new Firestore path.
    """
    try:
        stats = get_dashboard_stats_cached(child_user_id=child_id)
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating child stats: {str(e)}",
        )


@router.post("/reload")
async def reload_knowledge_base():
    """
    Reload the knowledge base from PDFs.
    This endpoint allows refreshing the vector store with updated PDFs.
    
    Returns:
        Status message
    """
    try:
        result = qa_service.reload_knowledge_base()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading knowledge base: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "Parent Dashboard AI Assistant"
    }


@router.get("/pdfs")
async def list_pdfs():
    """
    List all PDF files in the knowledge base.
    
    Returns:
        List of PDF file information (name and size)
    """
    try:
        pdf_files = []
        
        if PDFS_DIR.exists():
            for pdf_path in PDFS_DIR.glob("*.pdf"):
                file_size = pdf_path.stat().st_size
                pdf_files.append({
                    "name": pdf_path.name,
                    "size": file_size
                })
        
        return {"files": pdf_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")


def process_pdf_background(filename: str):
    """
    Background task to process a PDF file.
    This runs asynchronously after the file is uploaded.
    """
    try:
        result = qa_service.add_single_pdf(filename)
        logger.info("Background processing completed for %s: %s", filename, result.get("status", "completed"))
    except Exception as e:
        logger.exception("Error processing PDF %s in background: %s", filename, e)


@router.post("/pdfs/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a PDF file to the knowledge base.
    The file is saved immediately and processed asynchronously in the background.
    
    Args:
        file: The PDF file to upload
        background_tasks: FastAPI background tasks for async processing (injected by FastAPI)
    
    Returns:
        Status message with filename (processing happens in background)
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save file
        file_path = PDFS_DIR / file.filename
        
        # If file already exists, add a number suffix
        counter = 1
        original_filename = file.filename
        while file_path.exists():
            name_without_ext = Path(original_filename).stem
            extension = Path(original_filename).suffix
            new_filename = f"{name_without_ext}_{counter}{extension}"
            file_path = PDFS_DIR / new_filename
            counter += 1
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size for response
        file_size = file_path.stat().st_size
        
        # Process PDF asynchronously in background (don't wait for completion)
        background_tasks.add_task(process_pdf_background, file_path.name)
        
        # Return immediately after saving file (before processing)
        return {
            "status": "PDF uploaded successfully",
            "filename": file_path.name,
            "size": file_size,
            "processing_status": "pending"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@router.put("/pdfs/update")
async def update_pdf_name(request: UpdatePdfRequest):
    """
    Rename/update a PDF file in the knowledge base.
    
    Args:
        request: UpdatePdfRequest containing old_name and new_name
    
    Returns:
        Status message
    """
    try:
        old_name = request.old_name
        new_name = request.new_name
        
        # Ensure new_name has .pdf extension
        if not new_name.lower().endswith('.pdf'):
            new_name = f"{new_name}.pdf"
        
        old_path = PDFS_DIR / old_name
        new_path = PDFS_DIR / new_name
        
        # Check if old file exists
        if not old_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file '{old_name}' not found")
        
        # Check if new name already exists
        if new_path.exists() and old_path != new_path:
            raise HTTPException(status_code=400, detail=f"PDF file '{new_name}' already exists")
        
        # Rename file
        old_path.rename(new_path)
        
        # Reload knowledge base to reflect changes
        try:
            qa_service.reload_knowledge_base()
        except Exception as reload_error:
            # Log error but don't fail the update
            logger.warning("Failed to reload knowledge base after update: %s", reload_error)
        
        return {
            "status": "PDF renamed successfully",
            "old_name": old_name,
            "new_name": new_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating PDF: {str(e)}")


@router.delete("/pdfs/delete")
async def delete_pdf(file_name: str):
    """
    Delete a PDF file from the knowledge base.
    
    Args:
        file_name: Name of the PDF file to delete (query parameter)
    
    Returns:
        Status message
    """
    try:
        file_path = PDFS_DIR / file_name
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file '{file_name}' not found")
        
        # Validate it's a PDF file
        if not file_path.suffix.lower() == '.pdf':
            raise HTTPException(status_code=400, detail="Only PDF files can be deleted")
        
        # Remove from vector store first (before deleting file)
        try:
            qa_service.remove_single_pdf(file_name)
        except Exception as remove_error:
            # Log error but continue with file deletion
            logger.warning("Failed to remove PDF from vector store: %s", remove_error)
        
        # Delete file from disk
        file_path.unlink()
        
        return {
            "status": "PDF deleted successfully",
            "filename": file_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

