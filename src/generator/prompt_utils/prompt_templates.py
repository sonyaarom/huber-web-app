# src/prompts/prompt_templates.py
"""
Unified prompt templates module with base template class and specialized prompt types.
This module uses inheritance to reduce code duplication and a factory pattern
to create the appropriate prompt type.
"""
from src.generator.prompt_utils.prompt_builder import ReusableChatPromptBuilder, PromptComponent, PromptConfiguration
from src.generator.prompt_utils.config import settings
import logging
from typing import Dict, Optional, List, Union

logger = logging.getLogger(__name__)

# Centralized Langfuse client
from langfuse import Langfuse

def get_langfuse_client():
    """Get or create a Langfuse client singleton"""
    if not hasattr(get_langfuse_client, "client"):
        get_langfuse_client.client = Langfuse(
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            host=settings.LANGFUSE_HOST
        )
    return get_langfuse_client.client


class BasePromptTemplate:
    """Base class for all prompt templates with common functionality"""
    
    def __init__(self):
        self.prompt_builder = None
        self._initialize_prompt_builder()
    
    def _get_component_config(self) -> Dict[str, Dict]:
        """
        Define the component configuration for this prompt template.
        Override in subclasses to customize components.
        
        Returns:
            Dict[str, Dict]: Configuration for each component
        """
        return {
            "starting_meta_tags": {
                "prompt_id": settings.langfuse_prompt_ids["starting_meta_tags"],
                "weight": 1.0,
                "required": True,
                "separator_after": False
            },
            "role": {
                "prompt_id": settings.langfuse_prompt_ids["role-basic-v1"],
                "weight": 1.0,
                "required": True,
                "separator_after": False
            },
            "task": {
                "prompt_id": settings.langfuse_prompt_ids["main-task-v2"],
                "weight": 0.8,
                "required": True,
                "separator_after": False
            },
            "question": {
                "prompt_id": settings.langfuse_prompt_ids["question-v1"],
                "weight": 0.6,
                "required": True,
                "separator_after": False
            },
            "context": {
                "prompt_id": settings.langfuse_prompt_ids["context-v1"],
                "weight": 0.5,
                "required": True,  # Default to required
                "separator_after": False
            },
            "ending_meta_tags": {
                "prompt_id": settings.langfuse_prompt_ids["ending_meta_tags"],
                "weight": 0.1,
                "required": True,
                "separator_after": False
            }
        }
    
    def _initialize_prompt_builder(self):
        """Initialize the prompt builder with components based on configuration"""
        try:
            # Get component configurations
            component_configs = self._get_component_config()
            
            # Create main required components
            role_config = PromptComponent(
                prompt_id=component_configs["role"]["prompt_id"],
                weight=component_configs["role"]["weight"],
                required=component_configs["role"]["required"],
                separator_after=component_configs["role"].get("separator_after", False)
            )
            
            task_config = PromptComponent(
                prompt_id=component_configs["task"]["prompt_id"],
                weight=component_configs["task"]["weight"],
                required=component_configs["task"]["required"],
                separator_after=component_configs["task"].get("separator_after", False)
            )
            
            question_config = PromptComponent(
                prompt_id=component_configs["question"]["prompt_id"],
                weight=component_configs["question"]["weight"],
                required=component_configs["question"]["required"],
                separator_after=component_configs["question"].get("separator_after", False)
            )
            
            context_config = PromptComponent(
                prompt_id=component_configs["context"]["prompt_id"],
                weight=component_configs["context"]["weight"],
                required=component_configs["context"]["required"],
                separator_after=component_configs["context"].get("separator_after", False)
            )
            
            # Create additional components
            additional_configs = {}
            for key, config in component_configs.items():
                if key not in ["role", "task", "question", "context"]:
                    additional_configs[key] = PromptComponent(
                        prompt_id=config["prompt_id"],
                        weight=config["weight"],
                        required=config["required"],
                        separator_after=config.get("separator_after", False)
                    )
            
            # Initialize and configure the prompt builder
            self.prompt_builder = ReusableChatPromptBuilder()
            self.prompt_builder.configure(
                PromptConfiguration.create(
                    role_config=role_config,
                    task_config=task_config,
                    question_config=question_config,
                    context_config=context_config,
                    additional_configs=additional_configs
                )
            )
            
        except Exception as e:
            logger.error(f"Error initializing prompt builder: {e}")
            raise
    
    def build_prompt(self, user_question: str, context: str = "") -> str:
        """
        Build a prompt with the given user question and context
        
        Args:
            user_question (str): The user's question
            context (str, optional): Context information. Defaults to empty string.
            
        Returns:
            str: The formatted prompt
        """
        if not self.prompt_builder:
            raise ValueError("Prompt builder not initialized")
            
        # Ensure context is never empty if required by this prompt template
        component_configs = self._get_component_config()
        if component_configs["context"]["required"] and not context:
            context = "No additional context available."
            
        try:
            # Try with keyword arguments first
            return self.prompt_builder.create_chat_prompt(
                question=user_question,
                context=context
            )
        except TypeError:
            # Fall back to positional arguments if keyword args fail
            return self.prompt_builder.create_chat_prompt(
                user_question,
                context
            )
            
    def create_chat_prompt(self, user_question: str, context: str = "") -> str:
        """Alias for build_prompt to maintain compatibility with existing code"""
        return self.build_prompt(user_question, context)


class BasePrompt(BasePromptTemplate):
    """Minimal prompt with basic role and task components"""
    
    def _get_component_config(self) -> Dict[str, Dict]:
        """Override to customize the base prompt configuration"""
        config = super()._get_component_config()
        
        # Make context optional for the base prompt
        config["context"]["required"] = False
        
        return config


class MediumPrompt(BasePromptTemplate):
    """Medium complexity prompt with style guidelines"""
    
    def _get_component_config(self) -> Dict[str, Dict]:
        """Override to customize the medium prompt configuration"""
        config = super()._get_component_config()
        
        # Add style component
        config["style"] = {
            "prompt_id": settings.langfuse_prompt_ids["style-guidelines-v1"],
            "weight": 0.6,
            "required": True,
            "separator_after": False
        }
        
        return config


class AdvancedPrompt(BasePromptTemplate):
    """Advanced prompt with all available components"""
    
    def _get_component_config(self) -> Dict[str, Dict]:
        """Override to customize the advanced prompt configuration"""
        config = super()._get_component_config()
        
        # Use advanced role instead of basic
        config["role"]["prompt_id"] = settings.langfuse_prompt_ids["role-base-v2"]
        
        # Add style component
        config["style"] = {
            "prompt_id": settings.langfuse_prompt_ids["style-guidelines-v1"],
            "weight": 0.6,
            "required": True,
            "separator_after": False
        }
        
        # Add examples component
        config["shot_examples"] = {
            "prompt_id": settings.langfuse_prompt_ids["shot-examples-v1"],
            "weight": 0.4,
            "required": True,
            "separator_after": False
        }
        
        # Add quality check component
        config["quality_check"] = {
            "prompt_id": settings.langfuse_prompt_ids["quality-check-v1"],
            "weight": 0.3,
            "required": True,
            "separator_after": False
        }
        
        return config


class PromptFactory:
    """Factory class for creating prompt instances"""
    
    @staticmethod
    def create_prompt(prompt_type: str = "advanced") -> BasePromptTemplate:
        """
        Create a prompt instance of the specified type
        
        Args:
            prompt_type (str): Type of prompt to create ('base', 'medium', or 'advanced')
            
        Returns:
            BasePromptTemplate: An instance of the requested prompt type
        """
        prompt_type = prompt_type.lower()
        
        if prompt_type == "base":
            return BasePrompt()
        elif prompt_type == "medium":
            return MediumPrompt()
        elif prompt_type == "advanced":
            return AdvancedPrompt()
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


# Convenience function to get a prompt builder
def get_prompt_builder(prompt_type: Optional[str] = None):
    """
    Get a prompt builder instance based on the requested type
    
    Args:
        prompt_type (str, optional): Type of prompt to create.
            Defaults to the value in settings.DEFAULT_PROMPT_TYPE.
    
    Returns:
        BasePromptTemplate: An instance of the requested prompt type
    """
    prompt_type = prompt_type or getattr(settings, 'DEFAULT_PROMPT_TYPE', 'advanced')
    return PromptFactory.create_prompt(prompt_type)


# # Usage example
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     # Test each prompt type
#     for prompt_type in ["base", "medium", "advanced"]:
#         prompt = PromptFactory.create_prompt(prompt_type)
        
#         # Test with context
#         result = prompt.build_prompt(
#             "What is the capital of France?", 
#             "France is a country in Europe."
#         )
        
#         print(f"\nGenerated {prompt_type.title()} Prompt:")
#         print("-" * 50)
#         print(result[:500] + "..." if len(result) > 500 else result)
#         print("-" * 50)