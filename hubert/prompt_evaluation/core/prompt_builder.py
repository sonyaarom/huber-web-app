from langfuse import Langfuse
from enum import Enum
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Dict
from functools import lru_cache

from hubert.config import settings

logger = logging.getLogger(__name__)

def get_langfuse_client():
    """Get the Langfuse client"""
    try:
        return Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host
        )
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return None
@dataclass
class BuilderConfig:
    """Configuration for the ReusableChatPromptBuilder"""
    separator_value: str = "\n--------------------------------\n"
    max_context_length: int = 2048
    template_cache_size: int = 100

@dataclass
class PromptComponent:
    """Component configuration and content for prompt building"""
    prompt_id: str
    weight: float
    content: str = ""
    required: bool = False
    separator_after: bool = False
    
class PromptStrategy(ABC):
    """Abstract base class for different prompt composition strategies"""
    @abstractmethod
    def compose(self, components: Dict[str, PromptComponent], separator_config: Optional[Dict] = None) -> str:
        pass

class WeightedPromptStrategy(PromptStrategy):
    """Composition strategy that considers component weights and separators"""
    def __init__(self, separator: str = "\n\n"):
        self.separator = separator

    def compose(self, components: Dict[str, PromptComponent], separator_config: Optional[Dict] = None) -> str:
        weighted_components = sorted(
            components.items(),
            key=lambda x: x[1].weight,
            reverse=True
        )
        
        result = []
        for name, comp in weighted_components:
            if comp.content:
                result.append(comp.content)
                if hasattr(comp, 'separator_after') and comp.separator_after:
                    result.append(separator_config.get('value', self.separator) if separator_config else self.separator)
        
        return "\n\n".join(filter(None, result))

@dataclass
class PromptConfiguration:
    """Configuration for prompt components with weights"""
    components: Dict[str, PromptComponent]
    
    @classmethod
    def create(cls, 
               role_config: PromptComponent,
               task_config: PromptComponent,
               question_config: PromptComponent,
               context_config: PromptComponent,
               additional_configs: Optional[Dict[str, PromptComponent]] = None) -> 'PromptConfiguration':
        """Factory method to create a configuration with standard and additional components"""
        components = {
            "role": role_config,
            "task": task_config,
            "question": question_config,
            "context": context_config
        }
        
        if additional_configs:
            components.update(additional_configs)
            
        return cls(components=components)

class ReusableChatPromptBuilder:
    """A prompt builder optimized for chat scenarios that reuses Langfuse templates"""
    
    def __init__(
        self, 
        strategy: PromptStrategy = WeightedPromptStrategy(),
        config: Optional[BuilderConfig] = None
    ):
        self.components: Dict[str, PromptComponent] = {}
        self.strategy = strategy
        self.config = config or BuilderConfig()
        self.separator_config = {"value": self.config.separator_value}
        self._prompt_config: Optional[PromptConfiguration] = None

    @lru_cache(maxsize=100)
    def _get_compiled_template(self, prompt_id: str) -> str:
        """Cache compiled templates to avoid repeated API calls"""
        try:
            langfuse_client = get_langfuse_client()
            if langfuse_client is None:
                logger.error(f"Langfuse client not available.")
            return langfuse_client.get_prompt(prompt_id).compile()
        except Exception as e:
            logger.error(f"Failed to compile template {prompt_id}: {e}")
            
            

    def _add_component(self, name: str, prompt_id: str, weight: float, required: bool) -> None:
        """Add a component with compiled template"""
        try:
            template = self._get_compiled_template(prompt_id)
            separator_after = self._prompt_config.components[name].separator_after
            self.components[name] = PromptComponent(
                prompt_id=prompt_id,
                content=template,
                weight=weight,
                required=required,
                separator_after=separator_after
            )
        except Exception as e:
            logger.error(f"Error adding component {name} with prompt ID {prompt_id}: {e}")
            raise

    def create_chat_prompt(
        self, 
        user_question: str, 
        context: Optional[str] = None
    ) -> str:
        """Create a new chat prompt with updated user input"""
        if not user_question.strip():
            raise ValueError("User question cannot be empty")
        
        if not self._prompt_config:
            raise ValueError("Builder must be configured before creating prompts")
            
        try:
            question_config = self._prompt_config.components["question"]
            context_config = self._prompt_config.components["context"]
            
            # Get and compile question template with the actual user question
            prompt = self._get_compiled_template(question_config.prompt_id)
            question_template = prompt.replace("{{user_question}}", user_question)
            
            # Get and compile context template if context is provided
            context_template = ""
            if context:
                if len(context) > self.config.max_context_length:
                    logger.warning(f"Context exceeds maximum length. Truncating to {self.config.max_context_length} characters")
                    context = context[:self.config.max_context_length]
                prompt = self._get_compiled_template(context_config.prompt_id)
                context_template = prompt.replace("{{context}}", context)
            
            # Update components with compiled content
            self.components["question"] = PromptComponent(
                prompt_id=question_config.prompt_id,
                content=question_template,
                weight=question_config.weight,
                required=question_config.required,
                separator_after=question_config.separator_after
            )
            
            self.components["context"] = PromptComponent(
                prompt_id=context_config.prompt_id,
                content=context_template,
                weight=context_config.weight,
                required=context_config.required,
                separator_after=context_config.separator_after
            )
            
            return self.build()
            
        except Exception as e:
            logger.error(f"Error creating chat prompt: {e}")
            raise

    def configure(self, config: PromptConfiguration) -> 'ReusableChatPromptBuilder':
        """Configure the prompt builder with component templates"""
        self._prompt_config = config
        self._initialize_static_components()
        return self
        
    def _initialize_static_components(self):
        """Initialize static components that don't change between chat messages"""
        if not self._prompt_config:
            raise ValueError("Builder must be configured before initialization")
            
        try:
            for name, component_config in self._prompt_config.components.items():
                if name not in ["question", "context"]:  # Question and context are handled dynamically
                    self._add_component(
                        name,
                        component_config.prompt_id,
                        component_config.weight,
                        component_config.required
                    )
                    
        except Exception as e:
            logger.error(f"Error initializing static components: {e}")
            raise
            
    def validate(self) -> bool:
        """Validate that all required components are present"""
        return all(
            comp.content for comp in self.components.values() 
            if comp.required
        )

    def build(self) -> str:
        """Build the final prompt using the current strategy"""
        missing_components = [
            name for name, comp in self.components.items()
            if comp.required and not comp.content
        ]
        
        if missing_components:
            raise ValueError(f"Missing required components: {missing_components}")
            
        return self.strategy.compose(self.components, self.separator_config)

