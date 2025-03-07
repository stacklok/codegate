"""empty message

Revision ID: 1ee1be2156f7
Revises: e4c05d7591a8, 4b81c45b5da6
Create Date: 2025-03-07 09:02:54.636452+00:00

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "1ee1be2156f7"
down_revision: Union[str, None] = ("e4c05d7591a8", "4b81c45b5da6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
