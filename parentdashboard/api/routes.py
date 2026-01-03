"""
API routes for Parent Dashboard.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
import os
import shutil
from parentdashboard.schemas.request import QuestionRequest, UpdatePdfRequest
from parentdashboard.schemas.response import AnswerResponse
from parentdashboard.services.qa_service import QAService
from parentdashboard.config import PDFS_DIR

# Initialize router
router = APIRouter(prefix="/parentdashboard", tags=["Parent Dashboard"])

# Initialize QA service (singleton pattern)
qa_service = QAService()


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
        result = qa_service.answer_question(request.question)
        return AnswerResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


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
        print(f"Background processing completed for {filename}: {result.get('status', 'completed')}")
    except Exception as e:
        print(f"Error processing PDF {filename} in background: {str(e)}")


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
            print(f"Warning: Failed to reload knowledge base after update: {str(reload_error)}")
        
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
            print(f"Warning: Failed to remove PDF from vector store: {str(remove_error)}")
        
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

