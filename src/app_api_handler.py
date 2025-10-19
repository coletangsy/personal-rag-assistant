import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
import logging
import os
import sys

# Add the parent directory to the path to import from rag_app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your RAG components
from rag_app.main import initialize_components, create_rag_agent
from langchain_core.messages import HumanMessage

# Import CONFIG separately to handle path issues
import json
def load_config():
    """Load configuration from config.json file."""
    # Try to find config.json in the project root
    config_paths = [
        "config.json",
        "../config.json",
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    ]
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Configuration loaded from {config_path}")
                return config
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    
    # If no config found, raise error
    raise FileNotFoundError("config.json not found in any expected location")

# Load configuration
try:
    CONFIG = load_config()
except Exception as e:
    print(f"❌ Could not load config: {e}")
    CONFIG = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Personal RAG Assistant API",
    description="A personal RAG (Retrieval-Augmented Generation) assistant API",
    version="0.1.0"
)

# Add CORS middleware for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store initialized components
rag_agent = None
is_initialized = False

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    retrieval_attempts: Optional[int] = None
    
class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class InitResponse(BaseModel):
    status: str
    message: str

# Session storage (in production, use Redis or similar)
sessions: Dict[str, Dict[str, Any]] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_agent, is_initialized
    try:
        logger.info("🚀 Initializing RAG system on startup...")
        
        # Change to the correct working directory
        original_cwd = os.getcwd()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_root)
        
        try:
            # Initialize all components
            llm, tools_dict, _ = initialize_components()
            
            # Create RAG agent
            rag_agent = create_rag_agent(llm, tools_dict)
            
            is_initialized = True
            logger.info("✅ RAG system initialized successfully")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        is_initialized = False

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy" if is_initialized else "not_ready",
        message="Personal RAG Assistant API is running",
        timestamp=datetime.now().isoformat()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready - RAG system not initialized")
    
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        timestamp=datetime.now().isoformat()
    )

@app.post("/initialize", response_model=InitResponse)
async def initialize_system():
    """Manually initialize or reinitialize the RAG system"""
    global rag_agent, is_initialized
    
    try:
        logger.info("🔄 Manual initialization requested...")
        
        # Change to the correct working directory
        original_cwd = os.getcwd()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_root)
        
        try:
            # Initialize all components
            llm, tools_dict, _ = initialize_components()
            
            # Create RAG agent
            rag_agent = create_rag_agent(llm, tools_dict)
            
            is_initialized = True
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        return InitResponse(
            status="success",
            message="RAG system initialized successfully"
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready - please initialize first")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        logger.info(f"📨 Received message: {request.message[:100]}...")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        
        # Process the message asynchronously
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            process_message,
            request.message,
            session_id
        )
        
        # Extract response
        final_answer = result['messages'][-1].content
        retrieval_attempts = result.get('retrieval_attempts', 0)
        
        # Store session data
        sessions[session_id] = {
            "last_message": request.message,
            "last_response": final_answer,
            "timestamp": datetime.now().isoformat(),
            "retrieval_attempts": retrieval_attempts
        }
        
        return ChatResponse(
            response=final_answer,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            retrieval_attempts=retrieval_attempts
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

def process_message(message: str, session_id: str) -> Dict[str, Any]:
    """Process a message through the RAG agent"""
    logger.info(f"🔄 Processing message for session: {session_id}")
    
    # Change to the correct working directory for processing
    original_cwd = os.getcwd()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    try:
        # Initialize state with the message
        messages = [HumanMessage(content=message)]
        
        # Invoke the RAG agent
        result = rag_agent.invoke({
            "messages": messages,
            "original_question": message,
            "retrieved_content": "",
            "ranker_evaluation": "",
            "retrieval_attempts": 0
        })
        
        return result
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive parts only)"""
    safe_config = {
        "llm": {
            "model": CONFIG.get("llm", {}).get("model", "Unknown"),
            "temperature": CONFIG.get("llm", {}).get("temperature", "Unknown")
        },
        "vector_store": {
            "collection_name": CONFIG.get("vector_store", {}).get("collection_name", "Unknown"),
            "embedding_model": CONFIG.get("vector_store", {}).get("embedding_model", "Unknown")
        },
        "retriever": {
            "search_type": CONFIG.get("retriever", {}).get("search_type", "Unknown"),
            "k": CONFIG.get("retriever", {}).get("k", "Unknown")
        }
    }
    return safe_config

# Lambda handler for AWS deployment (optional)
handler = Mangum(app)

# For local development
if __name__ == "__main__":
    uvicorn.run(
        "app_api_handler:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )