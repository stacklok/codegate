"""update matcher types

Revision ID: 5e5cd2288147
Revises: 0c3539f66339
Create Date: 2025-02-19 14:52:39.126196+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5e5cd2288147"
down_revision: Union[str, None] = "0c3539f66339"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    # Update the matcher types. We need to do this every time we change the matcher types.
    # in /muxing/models.py
    op.execute(
        """
        UPDATE muxes
        SET matcher_type = 'fim', matcher_blob = ''
        WHERE matcher_type = 'request_type_match' AND matcher_blob = 'fim';
        """
    )
    op.execute(
        """
        UPDATE muxes
        SET matcher_type = 'chat', matcher_blob = ''
        WHERE matcher_type = 'request_type_match' AND matcher_blob = 'chat';
        """
    )
    op.execute(
        """
        UPDATE muxes
        SET matcher_type = 'catch_all'
        WHERE matcher_type = 'filename_match' AND matcher_blob != '';
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
        SET matcher_blob = 'fim', matcher_type = 'request_type_match'
        WHERE matcher_type = 'fim';
        """
    )
    op.execute(
        """
        UPDATE muxes
        SET matcher_blob = 'chat', matcher_type = 'request_type_match'
        WHERE matcher_type = 'chat';
        """
    )
    op.execute(
        """
        UPDATE muxes
        SET matcher_type = 'filename_match', matcher_blob = 'catch_all'
        WHERE matcher_type = 'catch_all';
        """
    )

    # Finish transaction
    op.execute("COMMIT;")
