# Personal RAG Assistant architecture

Personal RAG Assistant is a Python retrieval-augmented question-answering service with a web/API entrypoint, retrieval manager, and agent orchestration.

## Runtime topology

- [src/rag_app/main.py](src/rag_app/main.py) loads `config.json`, initializes the model and retriever, and builds the LangGraph workflow.
- [src/app_api_handler.py](src/app_api_handler.py) owns the FastAPI service, request/response models, health checks, initialization, and in-memory sessions.
- [src/rag_app/retriever_manager.py](src/rag_app/retriever_manager.py) loads or creates a local Chroma collection from the configured PDF or Obsidian vault. [src/rag_app/agents.py](src/rag_app/agents.py) owns safety routing, tool retrieval, ranking, answer generation, and polishing.
- [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml), and [pyproject.toml](pyproject.toml) define the packaged runtime and dependencies.

## Request flow

A request enters the API or web interface and passes through the safety and assistant nodes. The assistant may answer directly or invoke retrieval; the ranker evaluates retrieved context and may request another retrieval attempt. The current continuation function caps retrieval at five attempts before the final processing node formats the response. The current README distinguishes the web UI from future caching, search, and analytics directions.

## Boundaries

The retriever owns document loading, local vector persistence, and similarity retrieval; the graph and agent functions own routing and synthesis. The model boundary uses `ChatGoogleGenerativeAI`; source loading also configures embeddings for the Chroma collection. Credentials, private source documents, and generated vector data stay outside tracked source. Prompt definitions remain with the agent code. Container configuration describes the packaged runtime and does not establish that a remote deployment has been verified.

## Maintenance

Update this file when retrieval, agent orchestration, the API boundary, container runtime, or data source changes.
