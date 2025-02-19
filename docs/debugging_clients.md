# Debugging Clients (extensions)

CodeGate supports [different clients](https://docs.codegate.ai/integrations/) (extensions installed in a code editor).

Sometimes, there may be issues in the interaction between the client and CodeGate. If CodeGate is receiving the request correctly from the client, forwarding the request to the provider (LLM), and receiving the response from the provider, then the issue is likely in the client. Most commonly, the issue is a difference in the response sent by CodeGate and the one expected by the client.

To debug issues like the one mentioned above, a straightforward approach is removing CodeGate from the middle. Try the request directly to the provider (LLM) and compare the response with the one received from CodeGate. The following subsections will guide you on how to do this for different clients.

## Continue

As a prerequisite, follow [Continue's guide to build from source](https://docs.codegate.ai/integrations/) and be able to run Continue in debug mode. Depending on whether the issue was in a FIM or a Chat request, follow the corresponding subsection.

### FIM

The raw responses for FIM can be seen in the function `streamSse`.

https://github.com/continuedev/continue/blob/b6436dd84978c348bba942cc16b428dcf4235ed7/core/llm/stream.ts#L73-L77

Add a `console.log` statement to print the raw response inside the for-loop:
```typescript
console.log('Raw stream data:', value);
```

Observe the differences between the response received from CodeGate and the one received directly from the provider.

Sample configuration for CodeGate:
```json
"tabAutocompleteModel": {
    "title": "CodeGate - Provider",
    "provider": "openai",
    "model": "<model>",
    "apiKey": "<insert-api-key-if-required>",
    "apiBase": "http://localhost:8989/<provider>"
}
```

Sample configuration calling the provider directly:
```json
"tabAutocompleteModel": {
    "title": "Provider",
    "provider": "openai",
    "model": "<model>",
    "apiKey": "<insert-api-key-if-required>",
    "apiBase": "<provider-url>"
}
```

Hopefully, there will be a difference in the response that will help you identify the issue.
