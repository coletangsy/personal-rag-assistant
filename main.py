import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated, Sequence

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Local imports
from retriever_manager import RetrieverManager
from agent_functions import (
    safety_validation_llm_bound,
    assistant_llm_bound,
    retrieve_data_bound,
    ranker_llm_bound,
    pr_processing_llm_bound,
    is_question_safe,
    should_continue,
    should_continue_retrieval
)


# Load environment variables
load_dotenv()


# Load configuration from external JSON file
def load_config():
    """Load configuration from config.json file"""
    config_path = "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded from config.json")
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing config.json: {e}")
        raise


# Load configuration
CONFIG = load_config()


class AgentState(TypedDict):
    """State definition for the RAG agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_question: str
    retrieved_content: str
    ranker_evaluation: str


def initialize_components():
    """Initialize all necessary components for the RAG system"""
    print("🚀 Initializing RAG System Components...")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=CONFIG["llm"]["model"],
        temperature=CONFIG["llm"]["temperature"]
    )
    print("✅ LLM initialized")
    
    # Initialize Retriever Manager
    retriever_manager = RetrieverManager(
        persist_directory=CONFIG["vector_store"]["persist_directory"],
        collection_name=CONFIG["vector_store"]["collection_name"],
        embedding_model=CONFIG["vector_store"]["embedding_model"]
    )
    print("✅ Retriever Manager initialized")
    
    # Initialize retriever - check which source to use
    pdf_path = CONFIG.get("pdf", {}).get("path")
    obsidian_path = CONFIG.get("obsidian", {}).get("path")
    
    try:
        retriever = retriever_manager.initialize_retriever(
            pdf_path=pdf_path,
            obsidian_path=obsidian_path,
            search_type=CONFIG["retriever"]["search_type"],
            k=CONFIG["retriever"]["k"]
        )
        print("✅ Retriever initialized")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        raise
    
    # Get retriever tool
    try:
        retriever_tool = retriever_manager.get_retriever_tool()
        print("✅ Retriever tool created")
    except Exception as e:
        print(f"❌ Failed to create retriever tool: {e}")
        raise
    
    # Bind tools to LLM
    tools = [retriever_tool]
    llm_with_tools = llm.bind_tools(tools)
    tools_dict = {tool.name: tool for tool in tools}
    
    return llm_with_tools, tools_dict, retriever_manager


def create_rag_agent(llm, tools_dict):
    """Create and compile the RAG agent graph with early safety validation"""
    print("🧩 Building Enhanced RAG Agent Graph with Early Safety...")
    
    # Create wrapper functions with bound parameters
    def safety_validation_llm_bound_wrapper(state: AgentState):
        return safety_validation_llm_bound(state, llm, CONFIG)
    
    def assistant_llm_bound_wrapper(state: AgentState):
        return assistant_llm_bound(state, llm, CONFIG)
    
    def retrieve_data_bound_wrapper(state: AgentState):
        return retrieve_data_bound(state, tools_dict)

    def ranker_llm_bound_wrapper(state: AgentState):
        return ranker_llm_bound(state, llm, CONFIG)

    def pr_processing_llm_bound_wrapper(state: AgentState):
        return pr_processing_llm_bound(state, llm, CONFIG)

    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("Safety_agent", safety_validation_llm_bound_wrapper)  
    graph.add_node("Assistant_agent", assistant_llm_bound_wrapper)
    graph.add_node("Retriever_agent", retrieve_data_bound_wrapper)
    graph.add_node("Ranker_agent", ranker_llm_bound_wrapper)
    graph.add_node("PR_agent", pr_processing_llm_bound_wrapper)

    graph.set_entry_point("Safety_agent")  # Start with safety validation

    # Safety -> (if safe) -> Assistant OR (if unsafe) -> PR
    graph.add_conditional_edges(
        "Safety_agent",
        is_question_safe,
        {True: "Assistant_agent", False: "PR_agent"}  # Safe questions go to assistant, unsafe go directly to PR
    )
    
    # Assistant -> Retriever (when tools are needed)
    graph.add_conditional_edges(
        "Assistant_agent",
        should_continue,
        {True: "Retriever_agent", False: "PR_agent"}  # Go to PR if no tools needed
    )
    
    # Retriever -> Ranker (always process retrieved data)
    graph.add_edge("Retriever_agent", "Ranker_agent")
    
    # Ranker -> (if results are bad) -> Retriever OR (if results are good) -> PR
    graph.add_conditional_edges(
        "Ranker_agent",
        should_continue_retrieval,
        {True: "Retriever_agent", False: "PR_agent"}
    )
    
    # PR -> END (final answer)
    graph.add_edge("PR_agent", END)
    
    rag_agent = graph.compile()
    print("✅ Enhanced RAG Agent with Early Safety compiled successfully")
    
    return rag_agent


def run_conversation(rag_agent):
    """Run the conversational interface"""
    print("\n" + "="*60)
    print("💬 RAG Conversation Started")
    print("Type 'exit', 'quit', or 'stop' to end the conversation")
    print("="*60)
    
    while True:
        print("\n" + "="*60)
        user_input = input("❓ What is your question: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'stop']:
            print("👋 Ending conversation...")
            break
        
        if not user_input:
            print("⚠️  Please enter a question")
            continue
        
        try:
            print(f"🔄 Processing question: '{user_input}'")
            # Initialize state with the original question
            messages = [HumanMessage(content=user_input)]
            result = rag_agent.invoke({
                "messages": messages,
                "original_question": user_input,
                "retrieved_content": "",
                "ranker_evaluation": ""
            })
            
            print("\n" + "="*60)
            print(f"🤖 Final Answer: {result['messages'][-1].content}")
            
        except Exception as e:
            print(f"❌ Error processing your question: {e}")
            print("Please try again with a different question.")


def main():
    """Main function to run the RAG application"""
    try:
        # Initialize all components
        llm, tools_dict, _ = initialize_components()
        
        # Create RAG agent
        rag_agent = create_rag_agent(llm, tools_dict)
        
        # Start conversation
        run_conversation(rag_agent)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        print("Please check your configuration and try again.")
    finally:
        print("\n🎯 RAG Application terminated")

if __name__ == "__main__":
    main()