#Manages prompt versioning and tracking

from hubert.prompt_evaluation.prompts.prompt_templates import BasePrompt, MediumPrompt, AdvancedPrompt, BasePromptTemplate

class PromptManager:
    def __init__(self):
        self.prompts = {
            "base": BasePrompt(),
            "medium": MediumPrompt(),
            "advanced": AdvancedPrompt()
        }
    
    def get_prompt(self, prompt_name: str) -> BasePromptTemplate:
        return self.prompts[prompt_name]
    
    def get_all_prompts(self) -> Dict[str, BasePromptTemplate]:
        return self.prompts
    
    def get_prompt_by_id(self, prompt_id: str) -> BasePromptTemplate:
        for prompt_name, prompt in self.prompts.items():
            if prompt.prompt_id == prompt_id:
                return prompt
        raise ValueError(f"Prompt with ID {prompt_id} not found")