import weaviate
from weaviate.embedded import EmbeddedOptions


def restore_storage_backup(backup_path: str, backup_name: str):
    client = weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(
            persistence_data_path="./weaviate_data",
            grpc_port=50052,
            additional_env_vars={
                "ENABLE_MODULES": "backup-filesystem",
                "BACKUP_FILESYSTEM_PATH": backup_path,
            },
        )
    )
    client.connect()
    try:
        client.backup.restore(backup_id=backup_name, backend="filesystem", wait_for_completion=True)
    except Exception as e:
        print(f"Failed to restore backup: {e}")
    finally:
        client.close()
