#Manages prompt versioning and tracking

from src.prompts.base_prompt import BasePrompt
from src.prompts.medium_prompt import MediumPrompt
from src.prompts.advanced_prompt import AdvancedPrompt

class PromptManager:
    def __init__(self):
        self.prompts = {
            "base": BasePrompt(),
            "medium": MediumPrompt(),
            "advanced": AdvancedPrompt()
        }
    
    def get_prompt(self, prompt_name: str) -> Prompt:
        return self.prompts[prompt_name]
    
    def get_all_prompts(self) -> Dict[str, Prompt]:
        return self.prompts
    
    def get_prompt_by_id(self, prompt_id: str) -> Prompt:
        for prompt_name, prompt in self.prompts.items():
            if prompt.prompt_id == prompt_id:
                return prompt
        raise ValueError(f"Prompt with ID {prompt_id} not found")