import os
import json
import re
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
    

def is_question_safe(state: AgentState) -> bool:
    """Check if the question is safe based on safety agent's evaluation"""
    last_message = state["messages"][-1]
    
    # Check if the last message contains the safety evaluation
    if hasattr(last_message, 'content') and last_message.content:
        content = last_message.content
        
        print(f"🔒 Safety agent output: '{content}'")
        
        # Look for the safety evaluation in the format "safe:true" or "safe:false"
        match = re.search(r'safe:(true|false)', content.lower())
        
        if match:
            is_safe = match.group(1) == 'true'
            print(f"🔒 Safety evaluation: safe={is_safe}")
            return is_safe
        else:
            print("⚠️  Could not find safety evaluation pattern in output")
    
    # Default behavior: if we can't parse the safety evaluation, assume it's unsafe
    print("⚠️  Could not parse safety evaluation, defaulting to unsafe")
    return False


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


def should_continue_retrieval(state: AgentState) -> bool:
    """Check if retrieval should continue based on ranker's boolean evaluation"""
    last_message = state["messages"][-1]
    
    # Check if the last message contains the ranker's evaluation
    if hasattr(last_message, 'content') and last_message.content:
        content = last_message.content
        
        print(f"🔍 Ranker agent output: '{content}'")
        
        # Look for the boolean evaluation in the format "acceptable:true" or "acceptable:false"
        match = re.search(r'acceptable:(true|false)', content.lower())
        
        if match:
            is_acceptable = match.group(1) == 'true'
            print(f"🔍 Ranker evaluation: acceptable={is_acceptable}")
            
            # If the answer is not acceptable, continue retrieval
            # If the answer is acceptable, stop retrieval and proceed to assistant
            return not is_acceptable
        else:
            print("⚠️  Could not find ranker evaluation pattern in output")
    
    # Default behavior: if we can't parse the ranker's output, continue retrieval
    print("⚠️  Could not parse ranker evaluation, defaulting to continue retrieval")
    return True


def call_llm(state: AgentState, llm, agent_prompt):
    """Call the LLM with the current state and system prompt"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=agent_prompt)] + messages
    
    print(f"🤖 Calling LLM with {len(messages)} messages")
    print(f"📝 System prompt: {agent_prompt[:100]}...")
    if messages and hasattr(messages[-1], 'content'):
        print(f"💬 Last user message: {messages[-1].content}")
    
    message = llm.invoke(messages)
    
    print(f"📤 LLM response: {message.content[:200]}...")
    return {"messages": [message]}


def retrieve_data(state: AgentState, tools_dict: dict):
    """Execute tool calls from the LLM's messages"""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    
    print(f"🛠️  Found {len(tool_calls)} tool calls")
    
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
            print(f"📄 Result preview: {str(result)[:300]}...")
        
        results.append(ToolMessage(
            tool_call_id=tool_call['id'],
            name=tool_name,
            content=str(result)
        ))
    
    print("✅ Tools executed successfully")
    return {"messages": results}


def pr_processing_llm_bound(state: AgentState, llm):
    """PR agent that processes the final answer with proper context"""
    print("🎯 PR Agent: Processing final answer")
    
    # Collect all relevant information from the conversation
    original_question = None
    retrieved_content = None
    ranker_evaluation = None
    
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            original_question = msg.content
        elif isinstance(msg, ToolMessage):
            retrieved_content = msg.content
        elif isinstance(msg, AIMessage) and "acceptable:" in msg.content.lower():
            ranker_evaluation = msg.content
    
    print(f"📝 PR Agent Context:")
    print(f"  - Original Question: {original_question}")
    print(f"  - Retrieved Content Length: {len(retrieved_content) if retrieved_content else 0}")
    print(f"  - Ranker Evaluation: {ranker_evaluation}")
    
    # Create a comprehensive prompt that includes all necessary context
    comprehensive_prompt = f"""You are a Public Relations and Finalization Agent. Your task is to ensure the final answer is polished and accessible.

USER'S ORIGINAL QUESTION: {original_question or "Unknown"}

RETRIEVED INFORMATION:
{retrieved_content or "No information was retrieved"}

RANKER EVALUATION: {ranker_evaluation or "No evaluation provided"}

INSTRUCTIONS:
1. Based on the retrieved information above, provide a clear, comprehensive answer to the user's question
2. Structure the answer in a clear, easy-to-understand format
3. Use proper paragraphs and formatting
4. Include bullet points or numbered lists when appropriate
5. Ensure the language is suitable for a general audience
6. Make the answer concise but comprehensive
7. If citations or references are included, format them clearly
8. Ensure the tone is professional, helpful, and accessible
9. ALWAYS ensure the final output is in ENGLISH, regardless of the input language
10. If any content is not in English, translate it to clear, natural English

CRITICAL: If there is no answer available or the content indicates that no information was found, provide a polite response such as:
- "I'm sorry, but I couldn't find any relevant information to answer your question in my knowledge base."
- "Unfortunately, I don't have enough information in my database to provide a complete answer to your question."
- "Based on my search, I wasn't able to find specific information that addresses your question."

Your output is the FINAL answer that will be shown to the user."""
    
    return call_llm(state, llm, comprehensive_prompt)


def create_rag_agent(llm, tools_dict):
    """Create and compile the RAG agent graph with early safety validation"""
    print("🧩 Building Enhanced RAG Agent Graph with Early Safety...")
    
    # Create stateful functions with bound parameters
    def assistant_llm_bound(state: AgentState):
        return call_llm(state, llm, CONFIG["assistant_prompt"])
    
    def retrieve_data_bound(state: AgentState):
        return retrieve_data(state, tools_dict)

    def ranker_llm_bound(state: AgentState):
        return call_llm(state, llm, CONFIG["ranker_prompt"])

    def safety_validation_llm_bound(state: AgentState):
        return call_llm(state, llm, CONFIG["safety_prompt"])

    def pr_processing_llm_bound_wrapper(state: AgentState):
        return pr_processing_llm_bound(state, llm)

    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("Safety_agent", safety_validation_llm_bound)  # Moved to start
    graph.add_node("Assistant_agent", assistant_llm_bound)
    graph.add_node("Retriever_agent", retrieve_data_bound)
    graph.add_node("Ranker_agent", ranker_llm_bound)
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
            messages = [HumanMessage(content=user_input)]
            result = rag_agent.invoke({"messages": messages})
            
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