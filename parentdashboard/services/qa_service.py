"""
QA Service module.
Connects RAG pipeline with LLM to answer questions.
"""
from typing import Dict, Optional

from parentdashboard.rag.rag_pipeline import RAGPipeline
from parentdashboard.ai.llm import GroqLLM
from parentdashboard.ai.prompt import build_prompt, get_system_prompt
from parentdashboard.services.service import get_child_summary


class QAService:
    """Service for answering questions using RAG + LLM."""
    
    def __init__(self):
        """Initialize QA service with RAG pipeline and LLM."""
        self.rag_pipeline = RAGPipeline()
        self.llm = GroqLLM()
        
        # Initialize RAG pipeline
        self.rag_pipeline.initialize()
    
    def answer_question(self, question: str, child_id: Optional[str] = None) -> Dict[str, str]:
        """
        Answer a question using RAG + LLM with Sinhala support.
        
        Args:
            question: User's question (can be in Sinhala or English)
            child_id: Optional child identifier for personalized context
        
        Returns:
            Dictionary with 'answer' key containing the response
        """
        # Retrieve relevant context from PDFs
        context_chunks = self.rag_pipeline.retrieve_context(question, top_k=5)
        
        # Build prompt (handles Sinhala/English detection and general knowledge supplementation)
        prompt = build_prompt(question, context_chunks)
        
        # Append personalized child summary if available
        child_summary_text = ""
        if child_id:
            try:
                summary = get_child_summary(child_id)
                child_summary_text = f"\n\n[Child Summary]\n{summary}"
            except Exception:
                # Fail silently for personalization to avoid breaking core QA
                child_summary_text = ""
        
        if child_summary_text:
            prompt = (
                f"{prompt}{child_summary_text}\n\n"
                "When the question is about this child's progress, activities, or performance: "
                "answer directly from the [Child Summary] above. Do NOT mention that this information "
                "is not in the PDFs or knowledge base—simply give a clear, helpful answer based on the child's data."
            )
        
        system_prompt = get_system_prompt()
        
        # Generate answer
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # Note: We no longer force "not available" message when no context
            # The LLM will use general knowledge and mention it's not from PDFs
            # This allows helpful answers even when PDFs don't fully cover the topic
            
            return {"answer": answer}
        
        except Exception as e:
            # Error message in both languages
            error_msg_en = f"I apologize, but I encountered an error while processing your question. Please try again later. Error: {str(e)}"
            error_msg_si = f"කණගාටුයි, නමුත් ඔබේ ප්‍රශ්නය සැකසීමේදී දෝෂයක් ඇති විය. කරුණාකර පසුව නැවත උත්සාහ කරන්න. දෝෂය: {str(e)}"
            
            # Detect language and respond accordingly
            from parentdashboard.ai.prompt import detect_language
            lang = detect_language(question)
            error_msg = error_msg_si if lang == 'sinhala' else error_msg_en
            
            return {"answer": error_msg}
    
    def reload_knowledge_base(self) -> Dict[str, str]:
        """
        Reload the knowledge base from PDFs.
        
        Returns:
            Dictionary with status message
        """
        try:
            self.rag_pipeline.initialize(force_reload=True)
            return {"status": "Knowledge base reloaded successfully"}
        except Exception as e:
            return {"status": f"Error reloading knowledge base: {str(e)}"}
    
    def add_single_pdf(self, filename: str) -> Dict[str, str]:
        """
        Add a single PDF to the knowledge base without reloading everything.
        This is much faster than reloading the entire knowledge base.
        
        Args:
            filename: Name of the PDF file to add
        
        Returns:
            Dictionary with status message
        """
        try:
            self.rag_pipeline.add_single_pdf(filename)
            return {"status": f"PDF {filename} added to knowledge base successfully"}
        except Exception as e:
            return {"status": f"Error adding PDF {filename}: {str(e)}"}
    
    def remove_single_pdf(self, filename: str) -> Dict[str, str]:
        """
        Remove a single PDF from the knowledge base without reloading everything.
        This is much faster than reloading the entire knowledge base.
        
        Args:
            filename: Name of the PDF file to remove
        
        Returns:
            Dictionary with status message
        """
        try:
            self.rag_pipeline.remove_single_pdf(filename)
            return {"status": f"PDF {filename} removed from knowledge base successfully"}
        except Exception as e:
            return {"status": f"Error removing PDF {filename}: {str(e)}"}