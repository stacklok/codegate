"""add_on_delete_cascade

Revision ID: 4dec3e456c9e
Revises: e6227073183d
Create Date: 2025-01-21 08:20:12.221051+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4dec3e456c9e"
down_revision: Union[str, None] = "e6227073183d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # To add ON DELETE CASCADE to the foreign key constraint, we need to
    # rename the table, create a new table with the constraint, and copy
    # the data over.
    op.execute("ALTER TABLE prompts RENAME TO _prompts_old;")
    op.execute(
        """
        CREATE TABLE prompts (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            timestamp DATETIME NOT NULL,
            provider TEXT,       -- VARCHAR(255)
            request TEXT NOT NULL,  -- Record the full request that arrived to the server
            type TEXT NOT NULL, -- VARCHAR(50) (e.g. "fim", "chat")
            workspace_id TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );
        """
    )
    op.execute("INSERT INTO prompts SELECT * FROM _prompts_old;")
    op.execute("DROP TABLE _prompts_old;")

    # Doing the same for the sessions table
    op.execute("ALTER TABLE sessions RENAME TO _sessions_old;")
    op.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            active_workspace_id TEXT NOT NULL,
            last_update DATETIME NOT NULL,
            FOREIGN KEY (active_workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
        );
        """
    )
    op.execute("INSERT INTO sessions SELECT * FROM _sessions_old;")
    op.execute("DROP TABLE _sessions_old;")

    # Doing the same for the output table
    op.execute("ALTER TABLE outputs RENAME TO _outputs_old;")
    op.execute(
        """
        CREATE TABLE outputs (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            prompt_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            output TEXT NOT NULL,   -- Record the full response. If stream will be a list of objects
            FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
        );
        """
    )
    op.execute("INSERT INTO outputs SELECT * FROM _outputs_old;")
    op.execute("DROP TABLE _outputs_old;")

    # Doing the same for the alerts table
    op.execute("ALTER TABLE alerts RENAME TO _alerts_old;")
    op.execute(
        """
        CREATE TABLE alerts (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            prompt_id TEXT NOT NULL,
            code_snippet TEXT,
            trigger_string TEXT, -- VARCHAR(255)
            trigger_type TEXT NOT NULL,   -- VARCHAR(50)
            trigger_category TEXT,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
        );
        """
    )
    op.execute("INSERT INTO alerts SELECT * FROM _alerts_old;")
    op.execute("DROP TABLE _alerts_old;")

    # Dropping unused table
    op.execute("DROP TABLE settings;")

    # Create indexes for foreign keys
    op.execute("CREATE INDEX idx_outputs_prompt_id ON outputs(prompt_id);")
    op.execute("CREATE INDEX idx_alerts_prompt_id ON alerts(prompt_id);")
    op.execute("CREATE INDEX idx_prompts_workspace_id ON prompts (workspace_id);")
    op.execute("CREATE INDEX idx_sessions_workspace_id ON sessions (active_workspace_id);")


def downgrade() -> None:
    # Settings table
    op.execute(
        """
        CREATE TABLE settings (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            ip TEXT,             -- VARCHAR(45)
            port INTEGER,
            llm_model TEXT,      -- VARCHAR(255)
            system_prompt TEXT,
            other_settings TEXT  -- JSON stored as TEXT
        );
        """
    )
