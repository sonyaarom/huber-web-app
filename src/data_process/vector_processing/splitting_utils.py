import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
from logging import getLogger
from collections import defaultdict
import numpy as np


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

logger = getLogger(__name__)



def recursive_chunk_text(text: str, chunk_size: int, chunk_overlap: int, length_function = len, is_separator_regex = False) -> List[str]:
    """
    Splits text recursively into overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
    )

    chunks = text_splitter.split_text(text)

    return chunks


def character_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits text into overlapping chunks.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    
    chunks = text_splitter.split_text(text)

    return chunks


def semantic_chunk_text(text: str) -> List[str]:
    """
    Splits text based on semantic similarity.
    """
    text_splitter = SemanticChunker(OpenAIEmbeddings())

    chunks = text_splitter.split_text(text)

    return chunks



# def process_data_recursive(df: pd.DataFrame, chunk_lengths: List[int], embed_model: any, embed_model_name: str, base_path: str) -> List[Tuple[int, int, int, float, float]]:
#     """
#     Process text data into recursive chunks, apply BM25, embed the chunks, and save the results.
#     """
#     chunk_stats = []

#     for chunk_length in chunk_lengths:
#         chunk_overlap = 50 if chunk_length <= 256 else 200
#         logger.info(f"Processing chunks with max length: {chunk_length}")

#         all_chunks, unique_ids, general_ids, urls, last_updateds, html_contents = [], [], [], [], [], []

#         for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Chunking texts (Size: {chunk_length})"):
#             if isinstance(row['text'], str):
#                 chunks = recursive_chunk_text(row['text'], chunk_length, chunk_overlap)
#                 all_chunks.extend(chunks)
#                 unique_ids.extend([f"{row['id']}_{i+1}" for i in range(len(chunks))])
#                 general_ids.extend([row['id']] * len(chunks))
#                 urls.extend([row['url']] * len(chunks))
#                 last_updateds.extend([row.get('last_updated', '')] * len(chunks))
#                 html_contents.extend([row.get('html_content', '')] * len(chunks))

#         # Create DataFrame
#         chunked_df = pd.DataFrame({
#             'unique_id': unique_ids,
#             'url': urls,
#             'last_updated': last_updateds,
#             'html_content': html_contents,
#             'text': all_chunks,
#             'len': [len(chunk) for chunk in all_chunks],
#             'general_id': general_ids
#         })

#         # Remove short chunks
#         chunked_df = chunked_df[chunked_df['len'] >= 50]

#         logger.info(f"Created {len(chunked_df)} chunks for size {chunk_length}")

#         # Compute statistics
#         min_length = chunked_df['len'].min()
#         max_length = chunked_df['len'].max()
#         mean_length = chunked_df['len'].mean()
#         median_length = chunked_df['len'].median()
#         chunk_stats.append((chunk_length, min_length, max_length, mean_length, median_length))

#         logger.info(f"Chunk length stats (Size: {chunk_length}): Min={min_length}, Max={max_length}, Mean={mean_length:.2f}, Median={median_length:.2f}")

#         # Apply BM25 sparse vectorization


#         # Embed the text
#         # logger.info("Embedding text chunks...")
#         # chunked_df["embedding"] = chunked_df["text"].apply(lambda x: embed_model.encode(x).tolist())

#         # # Save results
#         # output_path = f"{base_path}/chunks_{chunk_length}_{embed_model_name}.json"
#         # with open(output_path, "w") as f:
#         #     json.dump(chunked_df.to_dict(orient="records"), f)

#         # logger.info(f"Saved chunked data to {output_path}")

#         # # Clear memory
#         # del chunked_df
#         # gc.collect()

#     return chunk_stats


# example_text = "Hello, how are you? I'm fine, thank you. How can I help you today? Here is a long text that should be split into chunks. I'm going to keep writing more and more text to test the chunking. This is the last sentence of the text."

# print(semantic_chunk_text(example_text))