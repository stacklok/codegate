"""add soft delete

Revision ID: 8e4b4b8d1a88
Revises: 5c2f3eee5f90
Create Date: 2025-01-20 14:08:40.851647

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8e4b4b8d1a88"
down_revision: Union[str, None] = "5c2f3eee5f90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE workspaces
        ADD COLUMN deleted_at DATETIME DEFAULT NULL;
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE workspaces DROP COLUMN deleted_at;")
