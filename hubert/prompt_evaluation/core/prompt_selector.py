import logging
from typing import List, Dict, Any
from hubert.prompt_evaluation.prompts.prompt_templates import PromptFactory, BasePromptTemplate

logger = logging.getLogger(__name__)

class PromptSelector:
    """
    Dynamically selects a prompt template based on the user's query.
    """
    def __init__(self, prompt_types: List[str] = None):
        """
        Initializes the PromptSelector.

        Args:
            prompt_types (List[str]): A list of prompt types to load.
                                      If None, loads 'base', 'medium', 'advanced'.
        """
        if prompt_types is None:
            prompt_types = ['base', 'medium', 'advanced']

        self.prompt_builders: Dict[str, BasePromptTemplate] = {}
        for p_type in prompt_types:
            try:
                self.prompt_builders[p_type] = PromptFactory.create_prompt(p_type)
                logger.info(f"Successfully loaded prompt builder for type: {p_type}")
            except Exception as e:
                logger.error(f"Failed to create prompt builder for type: {p_type}. Error: {e}")

    def select_prompt_builder(self, query: str) -> Any:
        """
        Selects a prompt builder based on simple rules applied to the query.

        Args:
            query (str): The user's query.

        Returns:
            An instance of a prompt builder, or None if no builders are available.
        """
        query_lower = query.lower()
        query_len = len(query.split())

        # Rule-based selection
        if 'advanced' in self.prompt_builders and (query_len > 10 or any(kw in query_lower for kw in ["compare", "explain", "how"])):
            logger.debug("Selected 'advanced' prompt template.")
            return self.prompt_builders['advanced']
        
        if 'base' in self.prompt_builders and 'what is' in query_lower:
            logger.debug("Selected 'base' prompt template.")
            return self.prompt_builders['base']
        
        if 'medium' in self.prompt_builders:
            logger.debug("Selected 'medium' prompt template for other queries.")
            return self.prompt_builders['medium']
            
        elif 'base' in self.prompt_builders:
            logger.debug("Selected 'base' prompt template as fallback.")
            return self.prompt_builders['base']
            
        else:
            # Fallback to the first available prompt builder
            if self.prompt_builders:
                default_builder_type = list(self.prompt_builders.keys())[0]
                logger.warning(f"No specific prompt rule matched. Falling back to '{default_builder_type}'.")
                return self.prompt_builders[default_builder_type]
            else:
                logger.error("No prompt builders available in PromptSelector.")
                return None 