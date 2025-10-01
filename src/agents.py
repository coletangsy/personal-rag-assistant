import re
from langchain_core.messages import SystemMessage, ToolMessage
from typing import Dict, Any

def call_llm(state, llm, agent_prompt):
    """
    Call the LLM with the current state and system prompt.
    
    Args:
        state: The current conversation state containing messages
        llm: The language model instance to invoke
        agent_prompt: The system prompt to prepend to the conversation
    
    Returns:
        dict: Updated state containing the LLM's response message
    """
    messages = list(state["messages"])
    messages = [SystemMessage(content=agent_prompt)] + messages
    
    print(f"🤖 Calling LLM with {len(messages)} messages")
    print(f"📝 System prompt: {agent_prompt[:100]}...")
    if messages and hasattr(messages[-1], 'content'):
        print(f"💬 Last user message: {messages[-1].content}")
    
    message = llm.invoke(messages)
    
    print(f"📤 LLM response: {message.content[:200]}...")
    return {"messages": [message]}


def is_question_safe(state):
    """
    Check if the question is safe based on safety agent's evaluation.
    
    Args:
        state: The current conversation state containing safety evaluation messages
    
    Returns:
        bool: True if the question is safe, False otherwise
    """
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


def should_continue(state):
    """
    Check if the last message contains tool calls.
    
    Args:
        state: The current conversation state containing messages
    
    Returns:
        bool: True if the last message contains tool calls, False otherwise
    """
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


def should_continue_retrieval(state):
    """
    Check if retrieval should continue based on ranker's evaluation and attempt count.
    
    Args:
        state: The current conversation state containing ranker evaluation and attempt count
    
    Returns:
        bool: True if retrieval should continue, False if maximum attempts reached or content is acceptable
    """
    # Use the stored ranker_evaluation from state instead of parsing from messages
    ranker_evaluation = state.get("ranker_evaluation", "")
    retrieval_attempts = state.get("retrieval_attempts", 0)
    
    print(f"🔍 Current retrieval attempt: {retrieval_attempts}")
    
    # Check if we've reached the maximum attempts (5)
    if retrieval_attempts >= 5:
        print("🛑 Maximum retrieval attempts (5) reached. Stopping retrieval loop.")
        return False
    
    if ranker_evaluation:
        print(f"🔍 Ranker agent output: '{ranker_evaluation}'")
        
        # Look for the boolean evaluation in the format "acceptable:true" or "acceptable:false"
        match = re.search(r'acceptable:(true|false)', ranker_evaluation.lower())
        
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


def safety_validation_llm_bound(state, llm, config):
    """
    Safety agent that evaluates if the question is safe using the original question from state.
    
    Args:
        state: The current conversation state containing the original question
        llm: The language model instance to invoke
        config: Configuration parameters for the safety agent
    
    Returns:
        dict: Updated state containing the safety evaluation message
    """
    print("🔒 Safety Agent: Evaluating question safety")
    
    # Use the stored original_question from state
    original_question = state.get("original_question", "")
    
    print(f"🔒 Safety agent checking question: '{original_question}'")
    
    # Create a safety evaluation prompt that uses the original question from state
    safety_prompt = f"""
    You are a Safety Evaluation Agent. Your task is to determine if the following user question is safe and appropriate to answer.

    USER'S QUESTION: {original_question}

    EVALUATION CRITERIA:
    - The question should not promote harm, violence, or illegal activities
    - The question should not contain hate speech or offensive content
    - The question should not request inappropriate or explicit content
    - The question should be appropriate for a general audience

    INSTRUCTIONS:
    1. Analyze the user's question above
    2. Determine if it meets the safety criteria
    3. Respond ONLY with: "safe:true" or "safe:false"
    4. Do not include any additional text or explanation

    Your response must be exactly one of these two options:
    - "safe:true" if the question is safe
    - "safe:false" if the question is unsafe
    """
    
    return call_llm(state, llm, safety_prompt)


def assistant_llm_bound(state, llm, config):
    """
    Assistant agent that generates search queries using the original question from state.
    
    Args:
        state: The current conversation state containing the original question
        llm: The language model instance to invoke
        config: Configuration parameters for the assistant agent
    
    Returns:
        dict: Updated state containing the assistant's response message
    """
    print("🤖 Assistant Agent: Generating search queries")
    
    # Use the stored original_question from state
    original_question = state.get("original_question", "")
    
    print(f"🤖 Assistant agent processing question: '{original_question}'")
    
    # Create an assistant prompt that uses the original question from state
    assistant_prompt = f"""
    You are an Assistant Agent. Your task is to help answer the user's question by generating appropriate search queries.

    USER'S QUESTION: {original_question}

    INSTRUCTIONS:
    1. Analyze the user's question above
    2. Generate search queries that would help retrieve relevant information
    3. Use the available retrieval tools to search for information
    4. If you need to search for information, use the retriever tool
    5. If you can answer directly without searching, provide the answer

    Remember to use the retrieval tools when needed to find the best information.
    """
    
    return call_llm(state, llm, assistant_prompt)


def retrieve_data_bound(state, tools_dict: Dict[str, Any]):
    """
    Execute tool calls from the LLM's messages and store retrieved content.
    
    Args:
        state: The current conversation state containing tool calls
        tools_dict: Dictionary mapping tool names to tool instances
    
    Returns:
        dict: Updated state containing tool execution results and retrieval metadata
    """
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
    
    # Increment retrieval attempts counter
    current_attempts = state.get("retrieval_attempts", 0)
    new_attempts = current_attempts + 1
    
    return {
        "messages": results, 
        "retrieved_content": str(results[0].content) if results else "",
        "retrieval_attempts": new_attempts  # Update the attempts counter
    }


def ranker_llm_bound(state, llm, config):
    """
    Ranker agent that evaluates the quality of retrieved content.
    
    Args:
        state: The current conversation state containing retrieved content and original question
        llm: The language model instance to invoke
        config: Configuration parameters for the ranker agent
    
    Returns:
        dict: Updated state containing the ranker evaluation message and stored evaluation
    """
    print("🔍 Ranker Agent: Evaluating retrieved content quality")
    
    # Use the stored retrieved_content from state
    retrieved_content = state.get("retrieved_content", "")
    original_question = state.get("original_question", "")
    
    print(f"🔍 Ranker agent evaluating content for question: '{original_question}'")
    print(f"🔍 Retrieved content length: {len(retrieved_content)}")
    
    # Create a ranker prompt that uses the original question and retrieved content from state
    ranker_prompt = f"""
    You are a Ranker Agent. Your task is to evaluate whether the retrieved information sufficiently answers the user's question.

    USER'S ORIGINAL QUESTION: {original_question}
    RETRIEVED INFORMATION: {retrieved_content}

    EVALUATION CRITERIA:
    - Does the retrieved information directly address the user's question?
    - Is the information comprehensive enough to provide a complete answer?
    - Is the information relevant and accurate?
    - Are there any gaps in the information that need additional retrieval?

    INSTRUCTIONS:
    1. Analyze the retrieved information against the user's question
    2. Determine if the information is acceptable for answering the question
    3. Respond ONLY with: "acceptable:true" or "acceptable:false"
    4. Do not include any additional text or explanation

    Your response must be exactly one of these two options:
    - "acceptable:true" if the retrieved information is sufficient
    - "acceptable:false" if more or better information is needed
    """
    
    result = call_llm(state, llm, ranker_prompt)
    # Store the ranker evaluation in state
    if result["messages"] and hasattr(result["messages"][-1], 'content'):
        result["ranker_evaluation"] = result["messages"][-1].content
    return result


def pr_processing_llm_bound(state, llm, config):
    """
    PR agent that processes the final answer with proper context.
    
    Args:
        state: The current conversation state containing all retrieval context
        llm: The language model instance to invoke
        config: Configuration parameters for the PR agent
    
    Returns:
        dict: Updated state containing the final polished answer for the user
    """
    print("🎯 PR Agent: Processing final answer")
    
    # Use the stored values from state instead of extracting from messages
    original_question = state.get("original_question", "Unknown")
    retrieved_content = state.get("retrieved_content", "No information was retrieved")
    ranker_evaluation = state.get("ranker_evaluation", "No evaluation provided")
    retrieval_attempts = state.get("retrieval_attempts", 0)
    
    print(f"📝 PR Agent Context:")
    print(f"  - Original Question: {original_question}")
    print(f"  - Retrieved Content Length: {len(retrieved_content) if retrieved_content else 0}")
    print(f"  - Ranker Evaluation: {ranker_evaluation}")
    print(f"  - Retrieval Attempts: {retrieval_attempts}")
    
    # Check if we reached the maximum attempts
    if retrieval_attempts >= 5:
        print("🔄 Adding maximum attempts reached context to PR agent")
        max_attempts_context = f"""
        IMPORTANT: After {retrieval_attempts} retrieval attempts, I was unable to find sufficient information to answer your question. 
        Please provide a polite response indicating that no relevant information was found after extensive searching.
        """
    else:
        max_attempts_context = ""
    
    # Create a comprehensive prompt that includes all necessary context
    comprehensive_prompt = f"""
    You are a Public Relations and Finalization Agent. Your task is to ensure the final answer is polished and accessible.

    USER'S ORIGINAL QUESTION: {original_question}
    RETRIEVED INFORMATION: {retrieved_content}
    RANKER EVALUATION: {ranker_evaluation}
    {max_attempts_context}

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

    Your output is the FINAL answer that will be shown to the user.
    """
    
    return call_llm(state, llm, comprehensive_prompt)