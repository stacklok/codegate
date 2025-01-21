"""init db

Revision ID: 30d0144e1a50
Revises:
Create Date: 2025-01-15 09:30:00.490697

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "30d0144e1a50"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Schema for codegate database using SQLite
    # Prompts table
    op.execute(
        """
        CREATE TABLE prompts (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            timestamp DATETIME NOT NULL,
            provider TEXT,       -- VARCHAR(255)
            request TEXT NOT NULL,  -- Record the full request that arrived to the server
            type TEXT NOT NULL -- VARCHAR(50) (e.g. "fim", "chat")
        );
        """
    )
    # Outputs table
    op.execute(
        """
        CREATE TABLE outputs (
            id TEXT PRIMARY KEY,  -- UUID stored as TEXT
            prompt_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            output TEXT NOT NULL,   -- Record the full response. If stream will be a list of objects
            FOREIGN KEY (prompt_id) REFERENCES prompts(id)
        );
        """
    )
    # Alerts table
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
            FOREIGN KEY (prompt_id) REFERENCES prompts(id)
        );
        """
    )
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
    # Create indexes for foreign keys and frequently queried columns
    op.execute("CREATE INDEX idx_outputs_prompt_id ON outputs(prompt_id);")
    op.execute("CREATE INDEX idx_alerts_prompt_id ON alerts(prompt_id);")
    op.execute("CREATE INDEX idx_prompts_timestamp ON prompts(timestamp);")
    op.execute("CREATE INDEX idx_outputs_timestamp ON outputs(timestamp);")
    op.execute("CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);")


def downgrade() -> None:
    pass
