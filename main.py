import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated, Sequence

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Local imports
from retriever_manager import RetrieverManager


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


def should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool calls"""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


def call_llm(state: AgentState, llm):
    """Call the LLM with the current state and system prompt"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=CONFIG["system_prompt"])] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


def retrieve_data(state: AgentState, tools_dict: dict):
    """Execute tool calls from the LLM's messages"""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_query = tool_call['args'].get('query', 'No query provided')
        
        print(f"🔧 Calling Tool: {tool_name} with query: {tool_query}")
        
        if tool_name not in tools_dict:
            print(f"⚠️  Tool '{tool_name}' does not exist")
            result = "Incorrect Tool Name. Please use available tools."
        else:
            result = tools_dict[tool_name].invoke(tool_query)
            print(f"📊 Result length: {len(str(result))} characters")
        
        results.append(ToolMessage(
            tool_call_id=tool_call['id'],
            name=tool_name,
            content=str(result)
        ))
    
    print("✅ Tools executed successfully")
    return {"messages": results}


def create_rag_agent(llm, tools_dict):
    """Create and compile the RAG agent graph"""
    print("🧩 Building RAG Agent Graph...")
    
    # Create stateful functions with bound parameters
    def call_llm_bound(state: AgentState):
        return call_llm(state, llm)
    
    def retrieve_data_bound(state: AgentState):
        return retrieve_data(state, tools_dict)
    
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("Assistant_agent", call_llm_bound)
    graph.add_node("Retriever_agent", retrieve_data_bound)
    graph.set_entry_point("Assistant_agent")
    
    graph.add_conditional_edges(
        "Assistant_agent",
        should_continue,
        {True: "Retriever_agent", False: END}
    )
    
    graph.add_edge("Retriever_agent", "Assistant_agent")
    
    rag_agent = graph.compile()
    print("✅ RAG Agent compiled successfully")
    
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
            messages = [HumanMessage(content=user_input)]
            result = rag_agent.invoke({"messages": messages})
            
            print("\n" + "="*60)
            print(f"🤖 Answer: {result['messages'][-1].content}")
            
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