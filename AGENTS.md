# Personal RAG Assistant repository guidance

## Documentation

- [Architecture](ARCHITECTURE.md) describes the API, retrieval, agent, and container boundaries.
- [Changelog](CHANGELOG.md) records meaningful changes.
- Keep implementation and deployment detail in README.md and update the architecture overview when the request flow or runtime changes.

## Working rules

- Keep secrets and local retrieval data outside Git.
- Preserve the bounded retrieval-attempt behavior and distinguish implemented features from future enhancements in README.md.
- Run the available Python checks before publishing application changes.

There is no committed automated test suite. A local syntax check can run without starting the model or document ingestion:

```sh
python3 -m compileall -q src
```

For documentation-only changes, check relative links and run `git diff --check`. A syntax check does not verify model access, retrieval quality, or remote deployment.
