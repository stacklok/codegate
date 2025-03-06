"""add provider_endpoint_name to muxes

Revision ID: 4b81c45b5da6
Revises: 769f09b6d992
Create Date: 2025-03-06 13:24:41.123857+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4b81c45b5da6"
down_revision: Union[str, None] = "769f09b6d992"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    # Add the new column
    op.execute(
        """
        ALTER TABLE muxes
        ADD COLUMN provider_endpoint_name TEXT;
        """
    )

    # Update the new column with data from provider_endpoints
    op.execute(
        """
        UPDATE muxes
        SET provider_endpoint_name = (
            SELECT name
            FROM provider_endpoints
            WHERE provider_endpoints.id = muxes.provider_endpoint_id
        );
        """
    )

    # Make the column NOT NULL after populating it
    # SQLite is funny about altering columns, so we actually need to clone &
    # swap the table
    op.execute("CREATE TABLE muxes_new AS SELECT * FROM muxes;")
    op.execute("DROP TABLE muxes;")
    op.execute(
        """
        CREATE TABLE muxes (
            id TEXT PRIMARY KEY,
            provider_endpoint_id TEXT NOT NULL,
            provider_model_name TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            matcher_type TEXT NOT NULL,
            matcher_blob TEXT NOT NULL,
            priority INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            provider_endpoint_type TEXT NOT NULL,
            provider_endpoint_name TEXT NOT NULL,
            FOREIGN KEY(provider_endpoint_id) REFERENCES provider_endpoints(id)
        );"""
    )
    op.execute("INSERT INTO muxes SELECT * FROM muxes_new;")
    op.execute("DROP TABLE muxes_new;")

    # Finish transaction
    op.execute("COMMIT;")


def downgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    try:
        # Check if the column exists
        op.execute(
            """
            SELECT provider_endpoint_name
            FROM muxes
            LIMIT 1;
            """
        )

        # Drop the column only if it exists
        op.execute(
            """
            ALTER TABLE muxes
            DROP COLUMN provider_endpoint_name;
            """
        )
    except Exception:
        # If there's an error (column doesn't exist), rollback and continue
        op.execute("ROLLBACK;")
        return

    # Finish transaction
    op.execute("COMMIT;")
