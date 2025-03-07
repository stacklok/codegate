"""add installation table

Revision ID: e4c05d7591a8
Revises: 3ec2b4ab569c
Create Date: 2025-03-05 21:26:19.034319+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e4c05d7591a8"
down_revision: Union[str, None] = "3ec2b4ab569c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("BEGIN TRANSACTION;")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS instance (
          id TEXT PRIMARY KEY,  -- UUID stored as TEXT
          created_at DATETIME NOT NULL
        );
        """
    )

    op.execute(
        """
        -- The following trigger prevents multiple insertions in the
        -- instance table. It is safe since the dimension of the table
        -- is fixed.

        CREATE TRIGGER single_instance
        BEFORE INSERT ON instance
        WHEN (SELECT COUNT(*) FROM instance) >= 1
        BEGIN
          SELECT RAISE(FAIL, 'only one instance!');
        END;
        """
    )

    # Finish transaction
    op.execute("COMMIT;")


def downgrade() -> None:
    op.execute("BEGIN TRANSACTION;")

    op.execute(
        """
        DROP TABLE instance;
        """
    )

    # Finish transaction
    op.execute("COMMIT;")
