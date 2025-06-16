"""Add tsvector

Revision ID: 1234567890
Revises: 20241216_add_entities
Create Date: 2024-12-18 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR


# revision identifiers, used by Alembic.
revision = '1234567890'
down_revision = '20241216_add_entities'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('page_embeddings_alpha', sa.Column('chunk_tsvector', TSVECTOR(), nullable=True))
    op.create_index('idx_chunk_tsvector', 'page_embeddings_alpha', ['chunk_tsvector'], unique=False, postgresql_using='gin')


def downgrade():
    op.drop_index('idx_chunk_tsvector', table_name='page_embeddings_alpha')
    op.drop_column('page_embeddings_alpha', 'chunk_tsvector') 