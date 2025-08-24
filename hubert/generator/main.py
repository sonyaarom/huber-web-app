from .generator_utils.generator_utils import initialize_models, generate_answer
from hubert.prompt_evaluation.prompts.prompt_templates import PromptFactory
from hubert.config import settings
from together import Together
import logging

logger = logging.getLogger(__name__)

TOGETHER_API_KEY = settings.together_api_key
client = Together(api_key=TOGETHER_API_KEY)

def main_generator(question: str, context: str, model_type: str = "openai"):
    import time
    
    # MODEL INITIALIZATION STAGE
    init_start = time.time()
    llm, embedding_generator, reranker_model = initialize_models(model_type=model_type)
    init_duration = time.time() - init_start
    logger.info(f"Model initialization took {init_duration:.3f} seconds.")

    # PROMPT BUILDING STAGE
    prompt_start = time.time()
    prompt_factory = PromptFactory().create_prompt(prompt_type=settings.default_prompt_type)
    prompt_text = prompt_factory.build_prompt(user_question=question, context=context)
    prompt_duration = time.time() - prompt_start
    logger.info(f"Prompt building took {prompt_duration:.3f} seconds.")

    # LLM GENERATION STAGE
    generation_start = time.time()
    text = generate_answer(llm=llm, 
                       generation_type='openai',
                       prompt_text=prompt_text, 
                       max_tokens=256, 
                       temperature=0.1)
    generation_duration = time.time() - generation_start
    logger.info(f"LLM generation took {generation_duration:.2f} seconds.")

    return text


def together_generator(question: str, context: str):
    import time
    
    # PROMPT BUILDING STAGE
    prompt_start = time.time()
    prompt_factory = PromptFactory().create_prompt(prompt_type=settings.default_prompt_type)
    prompt_text = prompt_factory.build_prompt(user_question=question, context=context)
    prompt_duration = time.time() - prompt_start
    logger.info(f"Prompt building took {prompt_duration:.3f} seconds.")
    logger.info(f"Prompt: {prompt_text}")

    # LLM GENERATION STAGE
    generation_start = time.time()
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt_text}],
    )
    generation_duration = time.time() - generation_start
    logger.info(f"LLM generation took {generation_duration:.2f} seconds.")
    
    return response.choices[0].message.content