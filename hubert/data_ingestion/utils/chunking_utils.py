from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from typing import List
import pandas as pd
import logging
from src.config import settings

OPENAI_API_KEY = settings.openai_api_key



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

