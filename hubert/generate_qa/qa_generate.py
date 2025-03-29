#qa_generate.py
import re  
import pandas as pd  
from config import settings
from llama_cpp import Llama
import pandas as pd  
from db_utils import fetch_page_content


# Fetch data and sample 150 rows
df = fetch_page_content()
df = df.sample(150)

# Path to the Llama model file
model_path = settings.model_path

# Initialize Llama model with optimized parameters for GPU acceleration
model = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # Use all available GPU layers
    n_ctx=4096,  # Context window size
    verbose=True,  # Enable verbose output
    logits_all=False,  # Disable computing logits for all tokens
    embedding=False,  # Disable embedding computation
    use_mmap=False,  # Disable memory mapping for better performance
    use_mlock=True,  # Lock memory to prevent swapping
    rope_scaling_type=1  # RoPE scaling type for position encoding
)

def generate_qa_pair(content):
    """
    Generate a question-answer pair from given content using Llama model
    Args:
        content: Input text content (can be string or dictionary)
    Returns:
        tuple: (question, answer) strings
    """
    # Handle dictionary content
    if isinstance(content, dict):
        content = content.get('text', str(content))
    
    # Convert to string and clean content
    content = str(content)
    content = content.replace('\n', ' ').replace('\r', ' ').strip()
    
    # Create prompt for the model
    prompt = (
        "Generate a question and answer pair based on the following content. "
        "Please provide your output in the following format: "
        "Question: <your question> Answer: <your answer> "
        f"Content: {content}"
    )
    
    # Generate response using the model
    response = model(
        prompt=prompt,
        max_tokens=150,  # Maximum length of generated response
        temperature=0.7,  # Controls randomness (higher = more random)
        top_p=0.95,  # Nucleus sampling parameter
        echo=False  # Don't include prompt in output
    )
    
    # Extract response text
    if isinstance(response, dict) and 'choices' in response:
        output_text = response['choices'][0]['text'].strip()
    else:
        output_text = str(response).strip()
    
    # Parse question and answer from response
    match = re.search(r"Question:\s*(.*?)\s*Answer:\s*(.*)", output_text, re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        # If parsing fails, return empty question and full output as answer
        question = ""
        answer = output_text
    
    return question, answer


if __name__ == "__main__":
    results = []
    # Process each row in the dataframe
    for index, row in df.iterrows():
        extracted_content = row['extracted_content']
    
        try:
            # Clean and prepare content for QA generation
            if isinstance(extracted_content, dict):
                cleaned_content = extracted_content.get('text', str(extracted_content))
            else:
                cleaned_content = str(extracted_content).replace('\n', ' ').strip()
            
            # Generate QA pair
            question, answer = generate_qa_pair(cleaned_content)
        except Exception as e:
            # Handle errors by logging and returning empty strings
            print(f"Error generating QA pair for id {row.get('id', index)}: {e}")
            question, answer = "", ""
        
        # Store results
        results.append({
            "id": row.get("id", index),
            "question": question,
            "answer": answer,
            "extracted_context": extracted_content
        })

    # Convert results to DataFrame and save to CSV
    qa_df = pd.DataFrame(results)
    qa_df.to_csv("qa_pairs.csv", index=False)
    print("QA pairs saved to qa_pairs.csv")
