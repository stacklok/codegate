"""add token usage columns

Revision ID: 0c3539f66339
Revises: 0f9b8edc8e46
Create Date: 2025-01-28 09:15:54.767311+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0c3539f66339"
down_revision: Union[str, None] = "0f9b8edc8e46"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    # We add the columns to the outputs table
    # Add the columns with default values to avoid issues with the existing data
    # The prices of the tokens may change in the future,
    # so we need to store the cost of the tokens at the time of the request
    op.execute("ALTER TABLE outputs ADD COLUMN input_tokens INT DEFAULT NULL;")
    op.execute("ALTER TABLE outputs ADD COLUMN output_tokens INT DEFAULT NULL;")
    op.execute("ALTER TABLE outputs ADD COLUMN input_cost FLOAT DEFAULT NULL;")
    op.execute("ALTER TABLE outputs ADD COLUMN output_cost FLOAT DEFAULT NULL;")

    # Finish transaction
    op.execute("COMMIT;")


def downgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    op.execute("ALTER TABLE outputs DROP COLUMN input_tokens;")
    op.execute("ALTER TABLE outputs DROP COLUMN output_tokens;")
    op.execute("ALTER TABLE outputs DROP COLUMN input_cost;")
    op.execute("ALTER TABLE outputs DROP COLUMN output_cost;")

    # Finish transaction
    op.execute("COMMIT;")
