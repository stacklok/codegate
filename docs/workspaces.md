# CodeGate Workspaces

Workspaces help you group related resources together. They can be used to organize your
configurations, muxing rules and custom prompts. It is important to note that workspaces
are not a tenancy concept; CodeGate assumes that it's serving a single user.

## Global vs Workspace resources

In CodeGate, resources can be either global (available across all workspaces) or workspace-specific:

- **Global resources**: These are shared across all workspaces and include provider endpoints,
  authentication configurations, and personas.
  
- **Workspace resources**: These are specific to a workspace and include custom instructions,
  muxing rules, and conversation history.

### Sessions and Active Workspaces

CodeGate uses the concept of "sessions" to track which workspace is active. A session represents
a user's interaction context with the system and maintains a reference to the active workspace.

- **Sessions**: Each session has an ID, an active workspace ID, and a last update timestamp.
- **Active workspace**: The workspace that is currently being used for processing requests.

Currently, the implementation expects only one active session at a time, meaning only one
workspace can be active. However, the underlying architecture is designed to potentially
support multiple concurrent sessions in the future, which would allow different contexts
to have different active workspaces simultaneously.

When a workspace is activated, the session's active_workspace_id is updated to point to that
workspace, and the muxing registry is updated to use that workspace's rules for routing requests.

## Workspace Lifecycle

Workspaces in CodeGate follow a specific lifecycle:

1. **Creation**: Workspaces are created with a unique name and optional custom instructions and muxing rules.
2. **Activation**: A workspace can be activated, making it the current context for processing requests.
3. **Archiving**: Workspaces can be archived (soft-deleted) when no longer needed but might be used again.
4. **Recovery**: Archived workspaces can be recovered to make them available again.
5. **Deletion**: Archived workspaces can be permanently deleted (hard-deleted).

### Default Workspace

CodeGate includes a default workspace that cannot be deleted or archived. This workspace is used
when no other workspace is explicitly activated.

## Workspace Features

### Custom Instructions

Each workspace can have its own set of custom instructions that are applied to LLM requests.
These instructions can be used to customize the behavior of the LLM for specific use cases.

### Muxing Rules

Workspaces can define muxing rules that determine which provider and model to use for different
types of requests. Rules are evaluated in priority order (first rule in the list has highest priority).

### Token Usage Tracking

CodeGate tracks token usage per workspace, allowing you to monitor and analyze resource consumption
across different contexts or projects.

### Alerts and Monitoring

Workspaces maintain their own alert history, making it easier to monitor and respond to issues
within specific contexts.

## Developing

### When to use workspaces?

Consider using separate workspaces when:

- You need different custom instructions for different projects or use cases
- You want to route different types of requests to different models
- You need to track token usage separately for different projects
- You want to isolate alerts and monitoring for specific contexts
- You're experimenting with different configurations and want to switch between them easily

### When should a resource be global?

Resources should be global when:

- They need to be shared across multiple workspaces
- They represent infrastructure configuration rather than usage patterns
- They're related to provider connectivity rather than specific use cases
- They represent reusable components like personas that might be used in multiple contexts

### Exporting resources

Currently, CodeGate doesn't provide a built-in mechanism to export workspace configurations.
However, you can use the API to retrieve workspace configurations and save them externally:

```bash
# Get workspace configuration
curl -X GET http://localhost:8989/api/v1/workspaces/{workspace_name} -H "Content-Type: application/json"

# Get workspace muxing rules
curl -X GET http://localhost:8989/api/v1/workspaces/{workspace_name}/muxes -H "Content-Type: application/json"

# Get workspace custom instructions
curl -X GET http://localhost:8989/api/v1/workspaces/{workspace_name}/custom-instructions -H "Content-Type: application/json"
```
