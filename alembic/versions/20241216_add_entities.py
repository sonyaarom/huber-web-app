"""Add entities to page_content

Revision ID: 20241216_add_entities
Revises: 20241215_sync_supabase
Create Date: 2024-12-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20241216_add_entities'
down_revision = '20241215_sync_supabase'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('page_content', sa.Column('entities', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade():
    op.drop_column('page_content', 'entities') 