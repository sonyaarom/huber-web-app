"""Create HNSW index on page_embeddings_a

Revision ID: 0001_add_hnsw_index
Revises: 
Create Date: 2024-06-18 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0001_add_hnsw_index'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    with op.get_context().autocommit_block():
        op.execute("SET statement_timeout = '3600s'")
        op.create_index(
            'ix_page_embeddings_a_embedding',
            'page_embeddings_a',
            ['embedding'],
            unique=False,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        )


def downgrade():
    op.drop_index('ix_page_embeddings_a_embedding', table_name='page_embeddings_a') 