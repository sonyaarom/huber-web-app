from .generator_utils.generator_utils import initialize_models, generate_answer
from .prompt_utils.prompt_templates import PromptFactory
from .prompt_utils.config import settings



def main_generator(question: str, context: str, model_type: str = "openai"):
    llm, embedding_generator, reranker_model = initialize_models(model_type=model_type)

    prompt_factory = PromptFactory().create_prompt(prompt_type=settings.DEFAULT_PROMPT_TYPE)
    prompt_text = prompt_factory.build_prompt(user_question=question, context=context)

    text = generate_answer(llm=llm, 
                       generation_type='openai',
                       prompt_text=prompt_text, 
                       max_tokens=256, 
                       temperature=0.1)

    return text

