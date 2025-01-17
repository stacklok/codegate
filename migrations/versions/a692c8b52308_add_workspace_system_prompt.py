"""add_workspace_system_prompt

Revision ID: a692c8b52308
Revises: 5c2f3eee5f90
Create Date: 2025-01-17 16:33:58.464223

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a692c8b52308'
down_revision: Union[str, None] = '5c2f3eee5f90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add column to workspaces table
    op.execute("ALTER TABLE workspaces ADD COLUMN system_prompt TEXT DEFAULT NULL;")


def downgrade() -> None:
    op.execute("ALTER TABLE workspaces DROP COLUMN system_prompt;")
