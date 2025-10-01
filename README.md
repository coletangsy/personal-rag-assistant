# 🤖 Personal RAG Assistant

A sophisticated AI-powered personal assistant that combines Retrieval-Augmented Generation (RAG) with a multi-agent workflow to provide intelligent, context-aware responses based on personal knowledge bases including Obsidian vaults and PDF documents.

## 🆕 What's New

### Latest Updates (October 2025)

**🚀 Enhanced Multi-Agent RAG System**
- **Early Safety Validation**: Questions are now evaluated for safety before processing, ensuring appropriate content filtering
- **Content Quality Ranking**: Automatic evaluation of retrieved information quality with multi-attempt retrieval for better answers
- **Professional Response Formatting**: Final answers are now polished with clear structure, proper formatting, and accessibility improvements
- **Improved Error Handling**: More robust error recovery and user-friendly error messages

**🔧 Better Project Organization**
- **Cleaner Codebase**: Reorganized file structure for easier maintenance and future development
- **Simplified Setup**: All configuration files remain easily accessible while source code is properly organized
- **Better Developer Experience**: Clear separation between application logic, agents, and data management

**What This Means for You:**
- More reliable and safe responses to your questions
- Higher quality answers with better information filtering
- Professional-looking responses with proper formatting
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
- **Interactive Chat Interface**: User-friendly command-line interface with real-time feedback

### 🛠️ Technical Features
- **LangGraph Integration**: Multi-agent workflow orchestration
- **Google Gemini Models**: State-of-the-art LLM and embedding models
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Error Handling**: Robust error recovery and graceful degradation
- **Progress Indicators**: Real-time status updates during processing
- **UV Package Management**: Fast and reliable dependency management

## 📹 Demo

https://github.com/user-attachments/assets/02b3a7bf-f0b5-4d0b-bca6-fc1a02d70863




## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd personal-rag-assistant
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up your environment**
   Create a `.env` file:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
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
   ```bash
   uv run src/main.py
   ```

## 📁 Project Structure

```
personal-rag-assistant/
├── 📁 src/                    # Main source code
│   ├── __init__.py
│   ├── main.py               # Main application entry point
│   ├── agents.py             # All agent functions (safety, assistant, ranker, PR)
│   └── retriever_manager.py  # Vector database and document processing
├── 📁 data/                  # Data storage
│   ├── chroma.sqlite3        # Chroma vector database
│   ├── vector_indices/       # Vector index files
│   └── documents/            # Source documents (PDFs, etc.)
├── .venv/                    # Python virtual environment (uv)
├── .gitignore                # Git ignore rules
├── .python-version           # Python version specification
├── config.json               # Application configuration
├── pyproject.toml            # Project dependencies (uv)
├── uv.lock                   # Dependency lock file
├── README.md                 # Project documentation
└── .env                      # Environment variables
```

## 🎮 Usage

### Starting the Assistant
```bash
uv run src/main.py
```

### Dependencies Management
```bash
# Add new dependency
uv add package_name

# Update dependencies
uv sync
```

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
- [ ] **Web Search Integration**: Add real-time web search capabilities
- [ ] **GUI Interface**: Develop a web-based or desktop interface
- [ ] **Advanced Caching**: Implement response caching for frequently asked questions
- [ ] **Multi-modal Support**: Support for images and other media types
- [ ] **Advanced Analytics**: Performance monitoring and usage analytics
- [ ] **Plugin System**: Extensible architecture for custom agents and tools
