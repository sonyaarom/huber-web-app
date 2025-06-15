from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector # Import the Vector type

Base = declarative_base()

class PageRaw(Base):
    __tablename__ = 'page_raw'
    id = Column(Text, primary_key=True)
    url = Column(Text, nullable=False)
    last_updated = Column(DateTime(timezone=True))
    last_scraped = Column(DateTime(timezone=True), default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)

class PageContent(Base):
    __tablename__ = 'page_content'
    id = Column(Text, primary_key=True)
    url = Column(Text, nullable=False)
    html_content = Column(Text)
    extracted_title = Column(Text)
    extracted_content = Column(Text)
    is_active = Column(Boolean, default=True, nullable=False)
    last_updated = Column(DateTime(timezone=True))
    last_scraped = Column(DateTime(timezone=True), default=func.now())

class PageKeywords(Base):
    __tablename__ = 'page_keywords'
    id = Column(Text, primary_key=True)
    uid = Column(Text, nullable=False, index=True)
    last_modified = Column(DateTime(timezone=True))
    tokenized_text = Column(TSVECTOR)
    raw_text = Column(Text)
    last_scraped = Column(DateTime(timezone=True), default=func.now())
    __table_args__ = (
        Index('idx_tokenized_text', tokenized_text, postgresql_using='gin'),
    )

class PageEmbeddings(Base):
    __tablename__ = 'page_embeddings_a'
    id = Column(Integer, primary_key=True)
    split_id = Column(Integer, nullable=False)
    url = Column(Text)
    chunk_text = Column(Text)
    embedding = Column(Vector(1536), nullable=False)
    last_scraped = Column(DateTime(timezone=True), default=func.now())

# Additional embedding tables
class PageEmbeddingsAlpha(Base):
    __tablename__ = 'page_embeddings_alpha'
    id = Column(String, primary_key=True)
    split_id = Column(Integer, nullable=False)
    url = Column(Text)
    chunk_text = Column(Text)
    embedding = Column(Vector(1536), nullable=False)
    last_scraped = Column(DateTime(timezone=True), default=func.now())

class PageEmbeddingsGeneric(Base):
    __tablename__ = 'page_embeddings'
    id = Column(Text, primary_key=True)
    split_id = Column(Text, nullable=False)
    url = Column(Text)
    chunk_text = Column(Text)
    embedding_vector = Column(Vector(1536), nullable=False)
    last_scraped = Column(DateTime(timezone=True), default=func.now())

class FailedJob(Base):
    __tablename__ = 'failed_jobs'
    job_id = Column(Integer, primary_key=True)
    uid = Column(Text, nullable=False, index=True)
    job_type = Column(Text, nullable=False) 