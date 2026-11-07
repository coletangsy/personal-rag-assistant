# Personal RAG Assistant repository guidance

## Documentation

- [Architecture](ARCHITECTURE.md) describes the API, retrieval, agent, and container boundaries.
- [Changelog](CHANGELOG.md) records meaningful changes.
- Keep implementation and deployment detail in README.md and update the architecture overview when the request flow or runtime changes.

## Working rules

- Keep secrets and local retrieval data outside Git.
- Preserve the bounded retrieval-attempt behavior and distinguish implemented features from future enhancements in README.md.
- Run the available Python checks before publishing application changes.

The Python application has no committed automated test suite. Check its syntax without starting the model or document ingestion:

```sh
python3 -m compileall -q src
```

Run `node --test .github/tests/delete-merged-branch.node-test.mjs` for the repository automation's 22 safety tests.

For documentation-only changes, check relative links and run `git diff --check`. A syntax check does not verify model access, retrieval quality, or remote deployment.
