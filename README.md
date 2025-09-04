# 🤖 Personal Assistant Bot

A sophisticated AI-powered personal assistant that combines Retrieval-Augmented Generation (RAG) with a two-stage agent workflow to provide intelligent, context-aware responses based on my personal knowledge base.

## ✨ Features

### 🎯 Core Functionality
- **Two-Stage Agent Workflow**: Searcher agent retrieves relevant information, followed by Assistant agent crafting responses
- **RAG Integration**: Retrieval-Augmented Generation powered by Chroma vector database that combines retrieved knowledge with LLM capabilities
- **Obsidian Integration**: Seamlessly indexes and retrieves information from my Obsidian vault
- **Interactive Chat Interface**: User-friendly command-line interface with real-time feedback

### 🛠️ Technical Features
- **Multi-agents Support**: Gemini models for different tasks (search vs. response generation)
- **Modular Architecture**: Clean separation of concerns with dedicated modules for agents, and RAG
- **Error Handling**: Robust error recovery and graceful degradation
- **Progress Indicators**: Real-time status updates during processing

## 📹 Demo

https://github.com/user-attachments/assets/c8986719-b5c9-4c01-bd70-5232e03be297



## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager
- API keys for:
  - Google Gemini API (for embeddings)
  - Obsidian vault path

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd personal-assistant-bot
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and paths
   ```

4. **Set up your environment**
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
   ```

5. **Run the application**
   ```bash
   uv run python main.py
   ```

## 📁 Project Structure

```
personal-assistant-bot/
├── main.py               # Main application entry point
├── retriever_manager.py  # RAG database and document processing
├── pyproject.toml        # Project dependencies
├── .env                  # Environment variables (gitignored)
├── .gitignore            # Git ignore rules
└── chroma.sqlite3/       # Chroma vector store (gitignored)
```

## 🎮 Usage

### Starting the Bot
```bash
uv run python main.py
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

🧩 Building RAG Agent Graph...
✅ RAG Agent compiled successfully

============================================================
💬 RAG Conversation Started
Type 'exit', 'quit', or 'stop' to end the conversation
============================================================

❓ What is your question: 
👤 You: What is machine learning?
```

### Example Workflow
1. **User Query**: "What is machine learning?"
2. **Stage 1**: Searcher agent analyzes query and retrieve information from database
3. **Stage 2**: Assistant agent crafts response based on research
4. **Response**: Comprehensive answer combining database knowledge and LLM capabilities

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | API key for Google Gemini embeddings | ✅ |
| `OBSIDIAN_VAULT_PATH` | Path to your Obsidian vault | ✅ |

### Model Configuration

### Adding New Features
- [ ] **Retrieve Data Rating**: Add a rating for evaluating data quality
- [ ] **GUI Interface**: Develop a web-based or desktop graphical user interface
- [ ] **Web Search Integration**: Add real-time web search capabilities using Tavily APIs
- [ ] **Advanced Caching**: Implement response caching for frequently asked questions
- [ ] **Advanced Filtering**: Implement content filtering and safety mechanisms
- [ ] **Monitoring Dashboard**: Add performance monitoring and analytics
