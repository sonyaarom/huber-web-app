"""Sync with current Supabase schema

Revision ID: 20241215_sync_supabase
Revises: d48a5b99342e
Create Date: 2024-12-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR


# revision identifiers, used by Alembic.
revision = '20241215_sync_supabase'
down_revision = 'd48a5b99342e'
branch_labels = None
depends_on = None


def upgrade():
    # Drop and recreate all tables to match Supabase schema
    
    # Drop existing tables in dependency order
    op.drop_table('page_embeddings_a')
    op.drop_table('page_keywords')
    op.drop_table('page_content')
    op.drop_table('failed_jobs')
    op.drop_table('page_raw')
    
    # Create page_raw table with TEXT id (matching Supabase)
    op.create_table('page_raw',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_scraped', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create page_content table (matching Supabase)
    op.create_table('page_content',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('html_content', sa.Text(), nullable=True),
        sa.Column('extracted_title', sa.Text(), nullable=True),
        sa.Column('extracted_content', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_scraped', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create page_keywords table (matching Supabase)
    op.create_table('page_keywords',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('uid', sa.Text(), nullable=False),
        sa.Column('last_modified', sa.DateTime(timezone=True), nullable=True),
        sa.Column('tokenized_text', TSVECTOR(), nullable=True),
        sa.Column('raw_text', sa.Text(), nullable=True),
        sa.Column('last_scraped', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_page_keywords_uid', 'page_keywords', ['uid'], unique=False)
    op.create_index('idx_tokenized_text', 'page_keywords', ['tokenized_text'], unique=False, postgresql_using='gin')
    
    # Create page_embeddings_a table (matching Supabase)
    op.create_table('page_embeddings_a',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('split_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.Text(), nullable=True),
        sa.Column('chunk_text', sa.Text(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.Column('last_scraped', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create page_embeddings_alpha table (matching Supabase)
    op.create_table('page_embeddings_alpha',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('split_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.Text(), nullable=True),
        sa.Column('chunk_text', sa.Text(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.Column('last_scraped', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create page_embeddings table (matching Supabase)
    op.create_table('page_embeddings',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('split_id', sa.Text(), nullable=False),
        sa.Column('url', sa.Text(), nullable=True),
        sa.Column('chunk_text', sa.Text(), nullable=True),
        sa.Column('embedding_vector', Vector(1536), nullable=False),
        sa.Column('last_scraped', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create failed_jobs table (matching Supabase)
    op.create_table('failed_jobs',
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('uid', sa.Text(), nullable=False),
        sa.Column('job_type', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('job_id')
    )
    op.create_index('ix_failed_jobs_uid', 'failed_jobs', ['uid'], unique=False)


def downgrade():
    # Revert to previous schema
    op.drop_table('failed_jobs')
    op.drop_table('page_embeddings')
    op.drop_table('page_embeddings_alpha')
    op.drop_table('page_embeddings_a')
    op.drop_table('page_keywords')
    op.drop_table('page_content')
    op.drop_table('page_raw')
    
    # Recreate the original schema (from d48a5b99342e)
    op.create_table('failed_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uid', sa.String(length=32), nullable=False),
        sa.Column('job_type', sa.String(length=255), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('failed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_resolved', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_failed_jobs_uid'), 'failed_jobs', ['uid'], unique=False)
    
    op.create_table('page_content',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_raw_uid', sa.String(length=32), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('raw_html', sa.Text(), nullable=True),
        sa.Column('last_processed', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_page_content_page_raw_uid'), 'page_content', ['page_raw_uid'], unique=False)
    
    op.create_table('page_embeddings_a',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_content_id', sa.Integer(), nullable=False),
        sa.Column('chunk', sa.Text(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('page_keywords',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_content_id', sa.Integer(), nullable=False),
        sa.Column('keywords_tsvector', TSVECTOR(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_page_keywords_page_content_id'), 'page_keywords', ['page_content_id'], unique=False)
    op.create_index('idx_keywords_tsvector', 'page_keywords', ['keywords_tsvector'], unique=False, postgresql_using='gin')
    
    op.create_table('page_raw',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('uid', sa.String(length=32), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_scraped', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_page_raw_uid'), 'page_raw', ['uid'], unique=True) 