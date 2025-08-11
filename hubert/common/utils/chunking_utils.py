from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from typing import List, Callable
import pandas as pd
import logging
from hubert.config import settings

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


class ChunkingStrategy:
    """Simple adapter exposing a split_text API over a callable."""
    def __init__(self, split_callable: Callable[[str], List[str]]):
        self._split_callable = split_callable
    
    def split_text(self, text: str) -> List[str]:
        return self._split_callable(text)


def get_chunking_strategy(name: str, **options) -> ChunkingStrategy:
    """
    Return a chunking strategy object exposing split_text(text) based on name.
    Supported names: 'recursive_chunk_text', 'character_chunk_text', 'semantic_chunk_text'.
    """
    normalized = (name or '').strip().lower()
    chunk_size = int(options.get('chunk_size', 512))
    chunk_overlap = int(options.get('chunk_overlap', 50))

    if normalized in ('recursive', 'recursive_chunk_text'):
        return ChunkingStrategy(lambda txt: recursive_chunk_text(txt, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    if normalized in ('character', 'character_chunk_text'):
        return ChunkingStrategy(lambda txt: character_chunk_text(txt, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    if normalized in ('semantic', 'semantic_chunk_text'):
        return ChunkingStrategy(lambda txt: semantic_chunk_text(txt))

    # Fallback to recursive if unknown
    logging.getLogger(__name__).warning(f"Unknown chunking strategy '{name}', defaulting to recursive.")
    return ChunkingStrategy(lambda txt: recursive_chunk_text(txt, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

