# src/rag/llm.py
from typing import Dict, List, Optional, Union, Any, Literal
from hubert.prompt_evaluation.workflows.utils.model_utils import initialise_llama, initialise_openai
from hubert.prompt_evaluation.workflows.retriever_workflow import HybridRetriever
# Import the PromptFactory instead of specific prompt classes
from hubert.prompt_evaluation.prompts.prompt_templates import PromptFactory
import logging
import time
from hubert.config import settings

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        embedding_model_name: str = settings.embedding_model,
        reranker_model_name: str = settings.reranker_model,
        alpha: float = settings.bm25_alpha,
        bm25_path: str = settings.bm25_values,
        prompt_builder = None,  # Make this a generic type to avoid import
        prompt_type: str = None,  # New parameter to specify prompt type
        max_retries: int = 3,
        retry_delay: float = 1.0,
        default_llm_model: str = 'llama',  # Default model to use if not specified
        init_openai: bool = False,  # Whether to initialize OpenAI
    ):
        logger.info(f"Initializing RAG system with {default_llm_model} model...")
        
        self.default_model_type = default_llm_model.lower()
        self.model_path = settings.MODEL_PATH
        
        # If prompt_builder is not provided but prompt_type is, create a prompt builder
        if prompt_builder is None and prompt_type is not None:
            try:
                prompt_builder = PromptFactory.create_prompt(prompt_type)
                logger.info(f"Created {prompt_type} prompt builder")
            except Exception as e:
                logger.error(f"Failed to create {prompt_type} prompt builder: {e}")
                prompt_builder = None
        
        # Store the prompt_builder directly
        self.prompt_builder = prompt_builder
        
        # Initialize models
        self.llm_llama = None
        self.llm_openai = None
        
        # Try to initialize Llama model
        try:
            logger.info("Initializing Llama model...")
            self.llm_llama = initialise_llama(model_path=self.model_path)
            logger.info("Llama model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {str(e)}")
        
        # Initialize OpenAI if requested
        if init_openai:
            try:
                logger.info("Initializing OpenAI model...")
                self.llm_openai = initialise_openai(api_key=settings.openai_api_key)
                self.openai_model = getattr(settings, 'openai_model', "gpt-3.5-turbo")
                logger.info("OpenAI model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI model: {str(e)}")
        
        # Check if at least one model is available
        if not self.llm_llama and not self.llm_openai:
            logger.warning("No LLM models are available. The system will return error messages for generation requests.")
        
        try:
            self.retriever = HybridRetriever(
                embedding_model_name=embedding_model_name,
                reranker_model_name=reranker_model_name,
                alpha=alpha,
                bm25_path=bm25_path or settings.bm25_values
            )
            logger.info("Retriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._retry_count = 0
        
        logger.info("RAG system initialized successfully")

    def _handle_rate_limit(self, func, *args, **kwargs):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) and retry_count < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** retry_count)
                    logger.warning(f"Rate limited. Waiting {wait_time:.2f}s before retry {retry_count + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    raise
        raise Exception(f"Max retries ({self.max_retries}) exceeded")

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.1,
        stop: Optional[List[str]] = None,
        echo: bool = False,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            gen_start = time.time()
            effective_model = (model_type or self.default_model_type).lower()
            
            # Don't attempt to reformat the prompt - it should already be formatted
            formatted_prompt = prompt
            
            # Try to use the requested model, with fallback logic
            if effective_model == 'llama':
                if not self.llm_llama:
                    if self.llm_openai:
                        logger.warning("Llama model not available, falling back to OpenAI")
                        effective_model = 'openai'
                    else:
                        error_msg = "Llama model not available and no fallback models are initialized."
                        logger.error(error_msg)
                        return {
                            'choices': [{
                                'text': f"Error: {error_msg} Unable to generate a response."
                            }]
                        }
                else:
                    # Use Llama model
                    response = self.llm_llama(
                        formatted_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop or ["<|eot_id|>"],
                        echo=echo
                    )
                    gen_time = time.time() - gen_start
                    logger.debug(f"Generation with Llama took {gen_time:.2f} seconds")
                    return response
            
            # If we're here, we're using OpenAI (either by choice or fallback)
            if effective_model == 'openai':
                if not self.llm_openai:
                    if self.llm_llama:
                        logger.warning("OpenAI model not available, falling back to Llama")
                        # Use Llama as fallback
                        response = self.llm_llama(
                            formatted_prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=stop or ["<|eot_id|>"],
                            echo=echo
                        )
                        gen_time = time.time() - gen_start
                        logger.debug(f"Generation with Llama (fallback) took {gen_time:.2f} seconds")
                        return response
                    else:
                        error_msg = "OpenAI model not available and no fallback models are initialized."
                        logger.error(error_msg)
                        return {
                            'choices': [{
                                'text': f"Error: {error_msg} Unable to generate a response."
                            }]
                        }
                else:
                    # Use OpenAI
                    openai_response = self.llm_openai.chat.completions.create(
                        model=self.openai_model,
                        messages=[{"role": "user", "content": formatted_prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop
                    )
                    response = {
                        'choices': [{
                            'text': openai_response.choices[0].message.content
                        }]
                    }
                    gen_time = time.time() - gen_start
                    logger.debug(f"Generation with OpenAI took {gen_time:.2f} seconds")
                    return response
            
            # If we get here, the model type is not supported
            error_msg = f"Unsupported model type: {effective_model}"
            logger.error(error_msg)
            return {
                'choices': [{
                    'text': f"Error: {error_msg}"
                }]
            }
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            # Return a dummy response instead of raising an error
            return {
                'choices': [{
                    'text': f"Error generating response: {str(e)}"
                }]
            }

    def retrieve(
        self,
        query: str,
        k: int = 10,
        use_reranking: bool = True,
        filter_dict: Optional[Dict] = None,
        return_only_ids: bool = False,
        return_all_chunk_ids: bool = False
    ) -> Union[List[Dict], List[str]]:
        try:
            return self.retriever.retrieve(
                query=query,
                k=k,
                use_reranking=use_reranking,
                filter_dict=filter_dict,
                return_only_ids=return_only_ids,
                return_all_chunk_ids=return_all_chunk_ids
            )
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            raise

    def get_texts_by_query(
        self,
        query: str,
        k: int = 3,
        use_reranking: bool = True,
        return_by_id: bool = False
    ) -> Union[List[str], Dict[str, List[str]]]:
        try:
            if return_by_id:
                return self.retriever.get_texts_by_top_k_general_ids(
                    query=query,
                    k=k,
                    use_reranking=use_reranking
                )
            else:
                return self.retriever.get_top_k_texts(
                    query=query,
                    k=k,
                    use_reranking=use_reranking
                )
        except Exception as e:
            logger.error(f"Error in get_texts_by_query: {str(e)}")
            raise

    def get_urls_by_query(
        self,
        query: str,
        k: int = 3,
        use_reranking: bool = True
    ) -> Dict[str, List[str]]:
        try:
            return self.retriever.get_urls_by_top_k_general_ids(
                query=query,
                k=k,
                use_reranking=use_reranking
            )
        except Exception as e:
            logger.error(f"Error in get_urls_by_query: {str(e)}")
            raise

    def rag_generate(
        self,
        query: str,
        system_prompt: str = "",
        k: int = 3,
        max_tokens: int = 128,
        temperature: float = 0.1,
        use_reranking: bool = True,
        llm_model: Optional[str] = None,
        retriever_type: Literal["parent", "initial"] = "initial"
    ) -> Dict:
        try:
            # First, retrieve the context based on retriever_type
            if retriever_type == "parent":
                # Get texts organized by parent/general IDs
                retrieved_texts_by_id = self.get_texts_by_query(
                    query=query,
                    k=k,
                    use_reranking=use_reranking,
                    return_by_id=True  # This will use get_texts_by_top_k_general_ids
                )
                
                # Flatten the texts from all parent IDs
                retrieved_texts = []
                for general_id, texts in retrieved_texts_by_id.items():
                    retrieved_texts.extend(texts)
                
                # Get URLs organized by parent/general IDs
                urls = self.get_urls_by_query(
                    query=query,
                    k=k,
                    use_reranking=use_reranking
                )
            else:  # "initial"
                # Get only the directly retrieved texts
                retrieved_texts = self.get_texts_by_query(
                    query=query,
                    k=k,
                    use_reranking=use_reranking,
                    return_by_id=False  # This will use get_top_k_texts
                )
                
                # Get URLs organized by parent/general IDs
                urls = self.get_urls_by_query(
                    query=query,
                    k=k,
                    use_reranking=use_reranking
                )
            
            # Create context from retrieved texts
            if retrieved_texts:
                context = "\n\n".join(retrieved_texts)
                
                # Truncate context if it's too large (especially for parent retriever)
                max_context_length = 2048  # Set a reasonable limit
                if len(context) > max_context_length:
                    logger.warning(f"Context length ({len(context)}) exceeds maximum ({max_context_length}). Truncating.")
                    context = context[:max_context_length]
            else:
                # Fallback for when retrieval returns nothing
                logger.warning("No texts retrieved, using fallback context")
                context = "No relevant information found in the database."
            
            # Store the context in the response for debugging
            response = {
                'context': context,
                'retriever_type': retriever_type
            }
            
            # Build the prompt using the prompt_builder if available
            if self.prompt_builder and hasattr(self.prompt_builder, 'create_chat_prompt'):
                try:
                    # First, try with keyword arguments
                    logger.debug(f"Building prompt for query: {query} with context length: {len(context)}")
                    defined_prompt = self.prompt_builder.create_chat_prompt(
                        user_question=query, 
                        context=context
                    )
                    logger.debug(f"Successfully built prompt with length: {len(defined_prompt)}")
                except TypeError as e:
                    logger.warning(f"Failed to build prompt with keyword args: {e}")
                    # If that fails, try with positional arguments
                    try:
                        defined_prompt = self.prompt_builder.create_chat_prompt(query, context)
                        logger.debug(f"Successfully built prompt with positional args, length: {len(defined_prompt)}")
                    except Exception as e2:
                        logger.error(f"Failed to build prompt with positional args too: {e2}")
                        # Fallback to default format
                        defined_prompt = f"{system_prompt}\n\n{query}\n\nContext: {context}"
                        logger.debug("Using fallback prompt format")
                except Exception as e:
                    logger.error(f"Unexpected error building prompt: {e}")
                    # Fallback to default format
                    defined_prompt = f"{system_prompt}\n\n{query}\n\nContext: {context}"
                    logger.debug("Using fallback prompt format due to unexpected error")
                
                # Add system prompt if not already handled by the prompt builder
                if system_prompt and system_prompt not in defined_prompt:
                    logger.debug(f"Adding system prompt: {system_prompt[:50]}...")
                    defined_prompt = f"{system_prompt}\n\n{defined_prompt}"
            else:
                # No prompt builder available, use default format
                logger.debug("No prompt builder available, using default format")
                defined_prompt = f"{system_prompt}\n\n{query}\n\nContext: {context}"
            
            # Store the defined prompt for debugging
            response['defined_prompt'] = defined_prompt
            
            # Generate response with the LLM
            effective_model = (llm_model or self.default_model_type).lower()
            logger.debug(f"Generating response with model: {effective_model}")
            
            # Check if Llama model is available before trying to generate
            if effective_model == 'llama' and not self.llm_llama:
                logger.error("Llama model not available for generation")
                response['choices'] = [{
                    'text': "Error: Llama model not available. Please check model initialization."
                }]
            else:
                # Try to generate response
                llm_response = self.generate_response(
                    prompt=defined_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_type=effective_model
                )
                
                # Combine responses
                response.update(llm_response)
            
            response['urls'] = urls
            response['retrieved_texts'] = retrieved_texts
            
            return response
        except Exception as e:
            logger.error(f"Error in rag_generate: {str(e)}")
            # Return a partial response with error information
            return {
                'error': str(e),
                'choices': [{
                    'text': f"Error in RAG generation: {str(e)}"
                }]
            }


def main():
    # Import the PromptFactory instead of specific prompt types
    from hubert.prompt_evaluation.prompts.prompt_templates import PromptFactory
    
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('src.core.prompt_builder').setLevel(logging.DEBUG)  # Enable verbose logging
    query = "Who is Stefan Lessmann?"

    try:
        print("\n=== Testing RAG System with Different Models and Retriever Types ===")
        
        # Create a single RAG system instance to reuse
        print("\n=== Initializing RAG System ===")
        prompt_builder = PromptFactory.create_prompt("base")
        print(f"Created base prompt builder")
        
        # Ask user which models to initialize
        init_llama = True  # Default to True for backward compatibility
        init_openai = input("Do you want to initialize OpenAI? (y/n, default: n): ").lower() == 'y'
        
        # Set the default model based on user preference
        if init_openai and not init_llama:
            default_model = 'openai'
        else:
            default_model = 'llama'
        
        # Initialize RAG system with the prompt builder - only once
        rag_system = RAGSystem(
            default_llm_model=default_model,
            prompt_builder=prompt_builder,
            init_openai=init_openai
        )
        
        print(f"Successfully initialized RAG system with default model: {default_model}")
        
        # Check which models are available
        if rag_system.llm_llama is None:
            print("Warning: Llama model failed to initialize.")
        else:
            print("Llama model is available.")
            
        if rag_system.llm_openai is None:
            print("Warning: OpenAI model is not initialized.")
        else:
            print("OpenAI model is available.")
        
        # Test with different retriever types and models
        for retriever_type in ["initial", "parent"]:
            # Determine which models to test
            models_to_test = []
            if rag_system.llm_llama is not None:
                models_to_test.append('llama')
            if rag_system.llm_openai is not None:
                models_to_test.append('openai')
            if not models_to_test:
                models_to_test = ['llama']  # Default to llama even if not available
            
            for model in models_to_test:
                print(f"\n\n=== Testing {model.upper()} with {retriever_type.capitalize()} Retriever ===")
                
                # Generate response
                response = rag_system.rag_generate(
                    query=query,
                    system_prompt="You are a helpful AI assistant.",
                    k=3,
                    max_tokens=256,
                    temperature=0.1,
                    llm_model=model,
                    retriever_type=retriever_type
                )
                
                print(f"\nQuery: {query}")
                print(f"Model: {model}")
                print(f"Retriever Type: {retriever_type}")
                
                # Print the context that was retrieved
                if 'context' in response:
                    print(f"Length of the context used: {len(response['context'])}")
                    print(f"Context preview: {response['context'][:150]}...")
                
                # Print a preview of the formatted prompt
                if 'defined_prompt' in response:
                    prompt_preview = response['defined_prompt'][:100] + "..." if len(response['defined_prompt']) > 100 else response['defined_prompt']
                    print(f"\nFormatted prompt preview:\n{prompt_preview}")
                
                print(f"\nResponse ({model.upper()} with {retriever_type.capitalize()} Retriever):")
                if response and 'choices' in response and response['choices']:
                    print(response['choices'][0]['text'])
                else:
                    print("No response generated")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()