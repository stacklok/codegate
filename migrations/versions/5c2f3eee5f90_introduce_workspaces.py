"""introduce workspaces

Revision ID: 5c2f3eee5f90
Revises: 30d0144e1a50
Create Date: 2025-01-15 19:27:08.230296

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5c2f3eee5f90"
down_revision: Union[str, None] = "30d0144e1a50"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Workspaces table
    op.execute(
        """
        CREATE TABLE workspaces (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            name TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT 0,
            UNIQUE (name)
        );
        """
    )
    op.execute("INSERT INTO workspaces (id, name, is_active) VALUES ('1', 'default', 1);")
    # Alter table prompts
    op.execute("ALTER TABLE prompts ADD COLUMN workspace_id TEXT REFERENCES workspaces(id);")
    op.execute("UPDATE prompts SET workspace_id = '1';")
    # Create index for workspace_id
    op.execute("CREATE INDEX idx_prompts_workspace_id ON prompts (workspace_id);")


def downgrade() -> None:
    # Drop the index for workspace_id
    op.execute("DROP INDEX IF EXISTS idx_prompts_workspace_id;")
    # Remove the workspace_id column from prompts table
    op.execute("ALTER TABLE prompts DROP COLUMN workspace_id;")
    # Drop the workspaces table
    op.execute("DROP TABLE IF EXISTS workspaces;")
