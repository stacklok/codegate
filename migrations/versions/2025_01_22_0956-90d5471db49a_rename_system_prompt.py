"""rename system_prompt

Revision ID: 90d5471db49a
Revises: 4dec3e456c9e
Create Date: 2025-01-22 09:56:21.520839+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "90d5471db49a"
down_revision: Union[str, None] = "4dec3e456c9e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE workspaces RENAME COLUMN system_prompt TO custom_instructions;")


def downgrade() -> None:
    op.execute("ALTER TABLE workspaces RENAME COLUMN custom_instructions TO system_prompt;")
