# 🤖 Personal RAG Assistant

Documentation: [working guide](AGENTS.md), [architecture](ARCHITECTURE.md) and [changelog](CHANGELOG.md).

A sophisticated AI-powered personal assistant that combines Retrieval-Augmented Generation (RAG) with a multi-agent workflow to provide intelligent, context-aware responses based on personal knowledge bases including Obsidian vaults and PDF documents.


## 📹 Demo
https://github.com/user-attachments/assets/dc462521-ccde-4a76-9b33-023ec03f2bb2


## 🆕 What's New

### Latest Updates (October 2025)

**🚀 Enhanced Multi-Agent RAG System**
- **Early Safety Validation**: Questions are now evaluated for safety before processing, ensuring appropriate content filtering
- **Content Quality Ranking**: Automatic evaluation of retrieved information quality with multi-attempt retrieval for better answers
- **Professional Response Formatting**: Final answers are now polished with clear structure, proper formatting, and accessibility improvements
- **Improved Error Handling**: More robust error recovery and user-friendly error messages

**🌐 Web API & Containerization**
- **FastAPI Web Interface**: Full REST API with session management and health checks
- **Docker Support**: Complete containerization with optimized Dockerfile and docker-compose
- **AWS Lambda Ready**: Mangum integration for serverless deployment
- **Interactive Web Chat**: HTML chat interface for browser-based interaction

**🔧 Better Project Organization**
- **Modular Architecture**: Clean separation between CLI and API interfaces
- **Enhanced Configuration**: Flexible config.json with multiple fallback paths
- **Better Developer Experience**: Clear separation between application logic, agents, and data management

**What This Means for You:**
- More reliable and safe responses to your questions
- Higher quality answers with better information filtering
- Professional-looking responses with proper formatting
- Multiple deployment options (CLI, Web API, Docker)
- Easier to customize and extend the system
- More stable and maintainable application


**Getting Started with the Latest Version:**
```bash
uv run src/main.py
```


## ✨ Features

### 🎯 Core Functionality
- **Multi-Agent Workflow**: Safety validation → Assistant → Retriever → Ranker → PR processing
- **RAG Integration**: Retrieval-Augmented Generation powered by Chroma vector database
- **Early Safety Validation**: Question safety evaluation before processing
- **Content Quality Ranking**: Automatic evaluation of retrieved content quality
- **Obsidian & PDF Integration**: Seamlessly indexes and retrieves information from Obsidian vaults and PDF documents
- **Interactive Interfaces**: Command-line interface and web API with real-time feedback

### 🛠️ Technical Features
- **LangGraph Integration**: Multi-agent workflow orchestration
- **Google Gemini Models**: State-of-the-art LLM and embedding models
- **FastAPI Web Server**: Modern async web framework with automatic docs
- **Docker Containerization**: Ready-to-deploy containerized application
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Error Handling**: Robust error recovery and graceful degradation
- **Progress Indicators**: Real-time status updates during processing
- **UV Package Management**: Fast and reliable dependency management


## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/coletangsy/personal-rag-assistant.git
   cd personal-rag-assistant
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up your environment**
   Create a `.env` file:
   ```env
   GOOGLE_API_KEY=<your_google_api_key_here>
   ```

4. **Configure your knowledge sources**
   Edit `config.json`:
   ```json
   {
    "llm": {
        "model": "gemini-2.5-flash",
        "temperature": 0
    },
    "vector_store": {
        "persist_directory": "./data/",
        "collection_name": "obsidian",
        "embedding_model": "models/gemini-embedding-001"
    },
    "pdf": {
        "path": ""
    },
    "obsidian": {
        "path": "/path/to/your/obsidian/vault"
    },
    "retriever": {
        "search_type": "similarity",
        "k": 3
    },
    "text_splitter": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
   }
   ```

5. **Run the application**
- **Option 1: Command Line Interface**
    ```bash
    uv run  -m src/rag_app/main.py
    ```

- **Option 2: Web API Server**
    ```bash
    uv run  src/app_api_handler.py
    ```

    Then visit: http://localhost:8000/docs for API documentation

- **Option 3: Docker Container**
    ```bash
    # Build and run with Docker
    docker build -t personal-rag-assistant .
    docker run -p 8000:8000 -e GOOGLE_API_KEY=<your_key> personal-rag-assistant

    # Or use docker-compose
    docker-compose up
    ```


## 📁 Project Structure

```
personal-rag-assistant/
├── 📁 src/                    # Main source code
│   ├── rag_app/              # Core RAG application
│   │   ├── main.py           # CLI application entry point
│   │   ├── agents.py         # All agent functions (safety, assistant, ranker, PR)
│   │   └── retriever_manager.py  # Vector database and document processing
│   ├── app_api_handler.py    # FastAPI web server and API endpoints
│   └── data/                 # Data storage (vector database, documents) ignored
│       ├── chroma.sqlite3    # Chroma vector database
│       ├── vector_indices/   # Vector index files
│       └── harrypotter.pdf   # Sample PDF document
├── chat_interface.html       # Web chat interface
├── config.json               # Application configuration
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Multi-container setup
├── pyproject.toml            # Project dependencies (uv)
├── uv.lock                   # Dependency lock file
├── .env                      # Environment variables
└── README.md                 # Project documentation
```


## 🎮 Usage

### Interactive Session Example
```
✅ Configuration loaded from config.json
🚀 Initializing RAG System Components...

✅ LLM initialized
✅ Retriever Manager initialized
✅ Retriever initialized
✅ Retriever tool created

🧩 Building Enhanced RAG Agent Graph with Early Safety...
✅ Enhanced RAG Agent with Early Safety compiled successfully

============================================================
💬 RAG Conversation Started
Type 'exit', 'quit', or 'stop' to end the conversation
============================================================

❓ What is your question: What is machine learning?
🔄 Processing question: 'What is machine learning?'
🔒 Safety Agent: Evaluating question safety
🔒 Safety agent checking question: 'What is machine learning?'
...
🤖 Final Answer: Machine learning is a subset of artificial intelligence...
```

### Agent Workflow
1. **Safety Agent**: Validates question safety before processing
2. **Assistant Agent**: Generates search queries for information retrieval
3. **Retriever Agent**: Executes searches in the knowledge base
4. **Ranker Agent**: Evaluates quality of retrieved content
5. **PR Agent**: Processes final answer with proper formatting and context


## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | API key for Google Gemini models | ✅ |

### Configuration Options
- **LLM Settings**: Model selection, temperature control
- **Vector Store**: Persistence directory, collection names, embedding models
- **Document Sources**: PDF and Obsidian vault paths
- **Retriever Settings**: Search type, result count (k)
- **Text Processing**: Chunk size and overlap for document splitting


## 🔄 Agent Workflow Details

### Safety Validation
- Early question safety evaluation
- Prevents processing of harmful or inappropriate content
- Configurable safety criteria

### Content Retrieval
- Automatic vector database initialization
- Support for multiple document formats (PDF, Markdown)
- Configurable search parameters

### Quality Assessment
- Automatic ranking of retrieved content
- Multi-attempt retrieval for better results
- Maximum attempt limits to prevent infinite loops

### Response Generation
- Final answer polishing and formatting
- Context-aware response generation
- Professional tone and accessibility


## Future Enhancements
- [X] **GUI Interface**: Develop a web-based interface
- [ ] **Advanced Caching**: Implement response caching for frequently asked questions
- [ ] **Web Search Integration**: Add real-time web search capabilities
- [ ] **Advanced Analytics**: Performance monitoring and usage analytics
