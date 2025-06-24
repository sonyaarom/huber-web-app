"""Add feedback and analytics tables

Revision ID: 0003
Revises: 0002
Create Date: 2025-01-09 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade():
    # Create user_feedback table for storing user feedback on answers
    op.create_table('user_feedback',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('generated_answer', sa.Text(), nullable=False),
        sa.Column('prompt_used', sa.String(255), nullable=True),
        sa.Column('retrieval_method', sa.String(255), nullable=True),
        sa.Column('sources_urls', postgresql.JSON(), nullable=True),
        sa.Column('rating', sa.String(50), nullable=False),  # 'positive' or 'negative'
        sa.Column('feedback_comment', sa.Text(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create query_analytics table for tracking all queries
    op.create_table('query_analytics',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('query_tokens', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('query_length', sa.Integer(), nullable=True),
        sa.Column('has_answer', sa.Boolean(), default=True, nullable=False),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('retrieval_method', sa.String(255), nullable=True),
        sa.Column('num_sources_found', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create retrieval_analytics table for MRR calculations  
    op.create_table('retrieval_analytics',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('query_analytics_id', sa.Integer(), nullable=False),
        sa.Column('retrieved_url', sa.Text(), nullable=False),
        sa.Column('rank_position', sa.Integer(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=True),
        sa.Column('is_relevant', sa.Boolean(), nullable=True),  # For future relevance judgments
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['query_analytics_id'], ['query_analytics.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Add indexes for performance
    op.create_index('idx_user_feedback_timestamp', 'user_feedback', ['timestamp'])
    op.create_index('idx_user_feedback_rating', 'user_feedback', ['rating'])
    op.create_index('idx_user_feedback_user_id', 'user_feedback', ['user_id'])
    
    op.create_index('idx_query_analytics_timestamp', 'query_analytics', ['timestamp'])
    op.create_index('idx_query_analytics_user_id', 'query_analytics', ['user_id'])
    # Note: gin_trgm_ops requires pg_trgm extension. For now, we'll use a basic btree index
    op.create_index('idx_query_analytics_query', 'query_analytics', ['query'])
    
    op.create_index('idx_retrieval_analytics_query_id', 'retrieval_analytics', ['query_analytics_id'])
    op.create_index('idx_retrieval_analytics_rank', 'retrieval_analytics', ['rank_position'])


def downgrade():
    # Drop indexes
    op.drop_index('idx_retrieval_analytics_rank')
    op.drop_index('idx_retrieval_analytics_query_id')
    op.drop_index('idx_query_analytics_query')
    op.drop_index('idx_query_analytics_user_id')
    op.drop_index('idx_query_analytics_timestamp')
    op.drop_index('idx_user_feedback_user_id')
    op.drop_index('idx_user_feedback_rating')
    op.drop_index('idx_user_feedback_timestamp')
    
    # Drop tables
    op.drop_table('retrieval_analytics')
    op.drop_table('query_analytics')
    op.drop_table('user_feedback') 