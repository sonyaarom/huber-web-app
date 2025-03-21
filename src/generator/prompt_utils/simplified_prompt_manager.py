"""
Simplified prompt manager that doesn't rely on external services.
This is used as a fallback when Langfuse or other external services are unavailable.
"""

import logging
import os
import time
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class SimplifiedPromptManager:
    """A simplified prompt manager that doesn't rely on external services."""
    
    def __init__(self, log_dir="logs/prompts", **kwargs):
        """
        Initialize the simplified prompt manager.
        
        Args:
            log_dir: Directory for logging prompts
            **kwargs: Additional arguments (ignored)
        """
        self.default_prompt_type = "fallback"
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info("Initialized SimplifiedPromptManager")
    
    def build_prompt(
        self, 
        user_question: str, 
        context: str = "", 
        prompt_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Build a simple prompt without relying on external services.
        
        Args:
            user_question: The user's question
            context: Retrieved context for the question
            prompt_type: Ignored in simplified version
            metadata: Additional metadata (logged but not used in prompt)
            
        Returns:
            Tuple[str, str]: The formatted prompt and the prompt type used
        """
        # Log the request
        self._log_prompt_request(user_question, context, metadata)
        
        # Simple fallback prompt
        prompt = (
            "You are a helpful assistant for Humboldt University. "
            "Please answer the following question based on the provided context.\n\n"
            f"Question: {user_question}\n\n"
            f"Context: {context[:1000] + '...' if len(context) > 1000 else context}"
        )
        
        return prompt, self.default_prompt_type
    
    def _log_prompt_request(
        self, 
        user_question: str, 
        context: str, 
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """
        Log the prompt request to a file.
        
        Args:
            user_question: The user's question
            context: Retrieved context
            metadata: Additional metadata
        """
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = os.path.join(self.log_dir, f"prompt_request_{timestamp}.log")
            
            with open(log_file, "w") as f:
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Question: {user_question}\n\n")
                f.write(f"Context: {context[:200]}...\n\n")
                
                if metadata:
                    f.write("Metadata:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")
        
        except Exception as e:
            logger.error(f"Error logging prompt request: {e}") 