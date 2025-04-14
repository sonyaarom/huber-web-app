from .generator_utils.generator_utils import initialize_models, generate_answer
from .prompt_utils.prompt_templates import PromptFactory
from .prompt_utils.config import settings
from together import Together
import logging

logger = logging.getLogger(__name__)

TOGETHER_API_KEY = settings.TOGETHER_API_KEY
client = Together()

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


def together_generator(question: str, context: str):

    prompt_factory = PromptFactory().create_prompt(prompt_type=settings.DEFAULT_PROMPT_TYPE)
    prompt_text = prompt_factory.build_prompt(user_question=question, context=context)
    logger.info(f"Prompt: {prompt_text}")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt_text}],
    )
    return response.choices[0].message.content


# if __name__ == "__main__":
#     question = "What is the capital of France?"
#     context = "France is a country in Western Europe. Its capital is Paris, which is known for its art, fashion, gastronomy and culture."
#     print(together_generator(question, context))