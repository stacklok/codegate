"""add persona table

Revision ID: 02b710eda156
Revises: 5e5cd2288147
Create Date: 2025-03-03 10:08:16.206617+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "02b710eda156"
down_revision: Union[str, None] = "5e5cd2288147"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    op.execute(
        """
            CREATE TABLE IF NOT EXISTS personas (
                id TEXT PRIMARY KEY,  -- UUID stored as TEXT
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                description_embedding BLOB NOT NULL
            );
            """
    )

    # Finish transaction
    op.execute("COMMIT;")


def downgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    op.execute(
        """
        DROP TABLE personas;
        """
    )

    # Finish transaction
    op.execute("COMMIT;")
