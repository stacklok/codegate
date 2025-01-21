"""merging system prompt and soft-deletes

Revision ID: e6227073183d
Revises: 8e4b4b8d1a88, a692c8b52308
Create Date: 2025-01-20 16:08:40.645298

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "e6227073183d"
down_revision: Union[str, None] = ("8e4b4b8d1a88", "a692c8b52308")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
