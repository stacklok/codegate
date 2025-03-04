"""migrate to glob pattern

Revision ID: 3ec2b4ab569c
Revises: 02b710eda156
Create Date: 2025-03-04 09:34:09.966863+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3ec2b4ab569c"
down_revision: Union[str, None] = "02b710eda156"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    # Update the matcher blobs to use glob patterns
    op.execute(
        """
        UPDATE muxes
        SET matcher_blob = '*' || matcher_blob
        WHERE matcher_type LIKE "%filename%" AND matcher_blob LIKE ".%"
        """
    )

    # Finish transaction
    op.execute("COMMIT;")


def downgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    op.execute(
        """
        UPDATE muxes
        SET matcher_blob = SUBSTRING(matcher_blob, 2)
        WHERE matcher_type LIKE "%filename%" AND matcher_blob LIKE "*%"
        """
    )

    # Finish transaction
    op.execute("COMMIT;")
