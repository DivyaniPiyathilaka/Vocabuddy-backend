"""
Request schemas for API endpoints.
"""
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., description="The parent's question", min_length=1, max_length=1000)
    child_id: str | None = Field(
        default=None,
        description="Optional child identifier for personalized answers",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How can I help my child with R sounds?",
                "child_id": "sUXK8GwJC6QNPQ7PCxSXEzT3TH63",
            }
        }


class UpdatePdfRequest(BaseModel):
    """Request model for updating/renaming a PDF file."""
    old_name: str = Field(..., description="Current name of the PDF file")
    new_name: str = Field(..., description="New name for the PDF file")
    
    class Config:
        json_schema_extra = {
            "example": {
                "old_name": "old_file.pdf",
                "new_name": "new_file.pdf"
            }
        }


class DeletePdfRequest(BaseModel):
    """Request model for deleting a PDF file."""
    file_name: str = Field(..., description="Name of the PDF file to delete")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "file_to_delete.pdf"
            }
        }
