"""add llm and mux

Revision ID: 0f9b8edc8e46
Revises: 90d5471db49a
Create Date: 2025-01-24 07:58:34.907908+00:00

"""

from typing import Sequence, Union

from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "0f9b8edc8e46"
down_revision: Union[str, None] = "90d5471db49a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with context.begin_transaction():
        # This table is used to store the models that are available
        # for references, e.g. in Muxing. The `auth_blob` field is
        # used to store the credentials for the model, which can be
        # a JSON object or a string, depending on the `auth_type`.
        # The `auth_type` field is used to determine how to interpret
        # the `auth_blob` field. If `auth_type` is `none`, then the
        # `auth_blob` field is ignored.
        # The `endpoint` field is used to store the endpoint of the
        # model.
        # NOTE: This resource is not namespaced by a workspace; that is
        # because the models are shared across workspaces.
        # NOTE: The lack of `deleted_at` is intentional. This resource
        # is not soft-deleted.
        # TODO: Do we need a display name here? An option is to
        # use the `name` field as the display name and normalize
        # the `name` field to be a slug when used as a reference.
        op.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,  -- UUID stored as TEXT
                provider TEXT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                endpoint TEXT NOT NULL DEFAULT '',
                auth_type TEXT NOT NULL DEFAULT 'none',
                auth_blob TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # This table is used to store the Muxing configuration. The
        # `destination_model_id` field is used to reference the model that the
        # Muxing configuration is for.
        # The `matcher_type` field is used to determine the type of the
        # matcher that is used in the Muxing configuration. e.g. `file_glob` would
        # be a matcher that uses file globbing to match files if a file is
        # detected in the prompt. The `matcher_blob` field is used to store the
        # configuration for the matcher, which can be a JSON object or a string,
        # depending on the `matcher_type`. On an initial implementation, the
        # `matcher_blob` field will simply be a string that is used to match
        # the prompt file name (if a file is detected in the prompt).
        # The `priority` field is used to determine the priority of the Muxing
        # configuration. The lower the number, the higher the priority. Note that
        # prompts will be matched against the Muxing configurations in ascending
        # order of priority.
        op.execute(
            """
            CREATE TABLE IF NOT EXISTS muxes (
                id TEXT PRIMARY KEY,  -- UUID stored as TEXT
                destination_model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                matcher_type TEXT NOT NULL,
                matcher_blob TEXT NOT NULL,
                priority INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # In terms of access patterns, the `muxes` table will be queried
        # to find the Muxing configuration for a given prompt. On initial search,
        # the `muxes` table will be queried by the `workspace_id`.
        op.execute("CREATE INDEX IF NOT EXISTS idx_muxes_workspace_id ON muxes (workspace_id);")


def downgrade() -> None:
    with context.begin_transaction():
        op.execute("DROP INDEX IF EXISTS idx_muxes_workspace_id;")
        op.execute("DROP TABLE IF EXISTS muxes;")
        op.execute("DROP TABLE IF EXISTS models;")
