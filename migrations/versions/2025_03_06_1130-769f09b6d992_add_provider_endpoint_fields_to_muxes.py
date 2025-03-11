"""add provider endpoint fields to muxes

Revision ID: 769f09b6d992
Revises: e4c05d7591a8
Create Date: 2025-03-06 11:30:11.647216+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "769f09b6d992"
down_revision: Union[str, None] = "e4c05d7591a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Begin transaction
    op.execute("BEGIN TRANSACTION;")

    # Add the new columns
    op.execute(
        """
        ALTER TABLE muxes
        ADD COLUMN provider_endpoint_type TEXT;
        """
    )
    op.execute(
        """
        ALTER TABLE muxes
        ADD COLUMN provider_endpoint_name TEXT;
        """
    )

    # Delete mux rules where provider_endpoint_id doesn't match a provider in
    # the database
    # This may seem extreme, but if the provider doesn't exist, the mux rule is
    # invalid and will error anyway if we try to use it, so we should prevent this invalid state from ever existing.
    # There is work on this branch to ensure that when a provider is deleted,
    # the associated mux rules are also deleted.
    op.execute(
        """
        DELETE FROM muxes 
        WHERE provider_endpoint_id NOT IN (SELECT id FROM provider_endpoints);
        """
    )

    # Update remaining rows with provider endpoint details
    op.execute(
        """
        UPDATE muxes
        SET
            provider_endpoint_type = (
                SELECT provider_type
                FROM provider_endpoints
                WHERE provider_endpoints.id = muxes.provider_endpoint_id
            ),
            provider_endpoint_name = (
                SELECT name
                FROM provider_endpoints
                WHERE provider_endpoints.id = muxes.provider_endpoint_id
            );
        """
    )

    # Make the columns NOT NULL after populating them
    # SQLite requires table recreation for this
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
        # Check if the columns exist
        op.execute(
            """
            SELECT provider_endpoint_type, provider_endpoint_name
            FROM muxes
            LIMIT 1;
            """
        )

        # Drop both columns if they exist
        op.execute(
            """
            ALTER TABLE muxes
            DROP COLUMN provider_endpoint_type;
            """
        )
        op.execute(
            """
            ALTER TABLE muxes
            DROP COLUMN provider_endpoint_name;
            """
        )
    except Exception:
        # If there's an error (columns don't exist), rollback and continue
        op.execute("ROLLBACK;")
        return

    # Finish transaction
    op.execute("COMMIT;")
