from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Index, LargeBinary, Float, ForeignKey
from sqlalchemy.dialects.postgresql import TSVECTOR, JSON, ARRAY
from flask_login import UserMixin
from sqlalchemy.orm import declarative_base, relationship
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
    __tablename__ = 'page_embeddings_alpha'
    id = Column(String, primary_key=True)
    split_id = Column(Integer, nullable=False)
    url = Column(Text)
    chunk_text = Column(Text)
    embedding = Column(Vector(1536), nullable=False)
    last_scraped = Column(DateTime(timezone=True), default=func.now())
    __table_args__ = (
        Index(
            'ix_page_embeddings_alpha_embedding',
            embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
    )

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

class User(UserMixin, Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(50), nullable=False, default='user')  # 'user' or 'admin'

    # Relationships
    feedback = relationship("UserFeedback", back_populates="user")
    queries = relationship("QueryAnalytics", back_populates="user")

    @property
    def is_admin(self):
        return self.role == 'admin'

    def __repr__(self):
        return f'<User {self.username}>'

class UserFeedback(Base):
    __tablename__ = 'user_feedback'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    query = Column(Text, nullable=False)
    generated_answer = Column(Text, nullable=False)
    prompt_used = Column(String(255), nullable=True)
    retrieval_method = Column(String(255), nullable=True)
    sources_urls = Column(JSON, nullable=True)
    rating = Column(String(50), nullable=False)  # 'positive' or 'negative'
    feedback_comment = Column(Text, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="feedback")

    def __repr__(self):
        return f'<UserFeedback {self.id}: {self.rating}>'

class QueryAnalytics(Base):
    __tablename__ = 'query_analytics'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    query = Column(Text, nullable=False)
    query_tokens = Column(ARRAY(String), nullable=True)
    query_length = Column(Integer, nullable=True)
    has_answer = Column(Boolean, default=True, nullable=False)
    response_time_ms = Column(Integer, nullable=True)
    retrieval_method = Column(String(255), nullable=True)
    num_sources_found = Column(Integer, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="queries")
    retrieval_results = relationship("RetrievalAnalytics", back_populates="query_analytics", cascade="all, delete-orphan")

    def __repr__(self):
        return f'<QueryAnalytics {self.id}: {self.query[:50]}...>'

class RetrievalAnalytics(Base):
    __tablename__ = 'retrieval_analytics'
    id = Column(Integer, primary_key=True)
    query_analytics_id = Column(Integer, ForeignKey('query_analytics.id', ondelete='CASCADE'), nullable=False)
    retrieved_url = Column(Text, nullable=False)
    rank_position = Column(Integer, nullable=False)
    similarity_score = Column(Float, nullable=True)
    is_relevant = Column(Boolean, nullable=True)  # For future relevance judgments
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    query_analytics = relationship("QueryAnalytics", back_populates="retrieval_results")

    def __repr__(self):
        return f'<RetrievalAnalytics {self.id}: Rank {self.rank_position}>'
