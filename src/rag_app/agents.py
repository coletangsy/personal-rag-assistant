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
        
        # Clean the evaluation string and make it more robust
        cleaned_evaluation = ranker_evaluation.strip().lower()
        
        # Look for the boolean evaluation with more flexible regex
        match = re.search(r'acceptable\s*:\s*(true|false)', cleaned_evaluation)
        
        if match:
            is_acceptable = match.group(1) == 'true'
            print(f"🔍 Ranker evaluation: acceptable={is_acceptable}")
            
            # If the answer is not acceptable, continue retrieval
            # If the answer is acceptable, stop retrieval and proceed to assistant
            return not is_acceptable
        else:
            print(f"⚠️  Could not find ranker evaluation pattern in output")
            print(f"⚠️  Raw output: '{ranker_evaluation}'")
            print(f"⚠️  Cleaned output: '{cleaned_evaluation}'")
            print(f"⚠️  Expected format: 'acceptable:true' or 'acceptable:false'")
    
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
    You are a Safety Evaluation Agent using ReAct (Reasoning and Acting) methodology.

    USER'S QUESTION: {original_question}

    TASK: Determine if this question is safe and appropriate to answer.

    Use this internal ReAct reasoning process (do not output this thinking):

    THOUGHT: I need to analyze this question for potential safety concerns systematically.

    REASONING: Evaluate the question against these safety criteria:

    1. HARM AND VIOLENCE: Check if the question promotes harmful activities, violence, or illegal actions
    2. HATE SPEECH AND OFFENSIVE CONTENT: Assess for discriminatory language or offensive material targeting individuals or groups
    3. INAPPROPRIATE CONTENT: Determine if it requests explicit, unethical, or inappropriate material
    4. SENSITIVE PERSONAL INFORMATION: Verify it doesn't seek confidential data like financial information, credentials, API keys, or personal identification details

    ACTION: Systematically evaluate each criterion and determine overall safety.

    OBSERVATION: Note any safety concerns or confirm the question is appropriate.

    CONCLUSION: Make final safety determination based on comprehensive analysis.

    IMPORTANT: Do ALL your thinking internally. Your response must be EXACTLY one of these two options:
    - "safe:true" if the question is safe
    - "safe:false" if the question is unsafe

    No additional text, explanation, or reasoning should be included in your output.
    """
    
    return call_llm(state, llm, safety_prompt)


def assistant_llm_bound(state, llm, config):
    """
    Assistant agent that generates search queries using ReAct reasoning pattern.
    """
    print("🤖 Assistant Agent: Generating search queries using ReAct pattern")
    
    original_question = state.get("original_question", "")
    print(f"🤖 Assistant agent processing question: '{original_question}'")
    
    # ReAct-based assistant prompt
    assistant_prompt = f"""
    You are an Assistant Agent using ReAct (Reasoning and Acting) methodology.

    USER'S QUESTION: {original_question}

    TASK: Help answer the user's question by determining the best approach to retrieve relevant information.

    Use this internal ReAct reasoning process (do not output this thinking):

    THOUGHT: I need to understand what the user is asking and determine the most effective strategy to help them.

    REASONING: Analyze the question systematically:

    1. QUESTION ANALYSIS: Identify the main topic, specific information needed, and user intent
    2. KNOWLEDGE ASSESSMENT: Determine if I can answer directly from general knowledge or need to search for specific information
    3. SEARCH STRATEGY: If retrieval is needed, identify key concepts, relevant keywords, and optimal search queries
    4. ACTION PLANNING: Decide whether to use retrieval tools with specific queries or provide a direct answer

    ACTION: Based on my analysis, execute the most appropriate approach - either use retrieval tools with well-crafted search queries or provide a direct comprehensive answer.

    OBSERVATION: Evaluate the chosen approach and proceed with implementation.

    CONCLUSION: Execute the planned action to best serve the user's information needs.

    AVAILABLE TOOLS:
    - retriever_tool: Use this to search for information in the knowledge base

    IMPORTANT: Do ALL your thinking internally. Based on your analysis:
    - If you need to search for information: Use the retriever_tool with appropriate search queries
    - If you can answer directly without needing specific information: Provide a comprehensive answer

    For this question about handling pain quotes, you should use the retriever_tool to search for relevant quotes and information.
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
    Ranker agent that evaluates retrieved content quality using ReAct reasoning pattern.
    """
    print("🔍 Ranker Agent: Evaluating retrieved content quality using ReAct pattern")
    
    retrieved_content = state.get("retrieved_content", "")
    original_question = state.get("original_question", "")
    retrieval_attempts = state.get("retrieval_attempts", 0)
    
    print(f"🔍 Ranker agent evaluating content for question: '{original_question}'")
    print(f"🔍 Retrieved content length: {len(retrieved_content)}")
    print(f"🔍 Current retrieval attempt: {retrieval_attempts}")
    
    # ReAct-based ranker prompt with stronger output constraints
    ranker_prompt = f"""
    You are a Ranker Agent using ReAct (Reasoning and Acting) methodology.

    USER'S ORIGINAL QUESTION: {original_question}
    RETRIEVED INFORMATION: {retrieved_content}
    CURRENT RETRIEVAL ATTEMPT: {retrieval_attempts} / 5

    TASK: Evaluate whether the retrieved information is sufficient to answer the user's question.

    Use this internal ReAct reasoning process (do not output this thinking):

    THOUGHT: I need to assess the quality and relevance of the retrieved information against the user's question to determine if additional retrieval is needed.

    REASONING: Systematically evaluate the retrieved content:

    1. RELEVANCE ANALYSIS: Assess how directly the content addresses the user's question, covering key concepts and matching user intent
    2. COMPLETENESS ASSESSMENT: Determine if there's enough information for a comprehensive answer and identify any critical gaps
    3. QUALITY EVALUATION: Check for accuracy, specificity, clarity, and absence of contradictions in the retrieved information
    4. RETRIEVAL CONTEXT: Consider current attempt number and whether continued searching would likely yield better results

    ACTION: Based on comprehensive evaluation, decide whether the current information is sufficient or if additional retrieval attempts would improve the answer quality.

    OBSERVATION: Analyze the overall sufficiency of available information considering both quality and retrieval attempt context.

    CONCLUSION: Make final determination on whether to proceed with current information or continue searching.

    CRITICAL OUTPUT REQUIREMENT:
    Your response must be EXACTLY one of these two options (copy exactly, no extra spaces, no punctuation, no additional text):

    acceptable:true

    OR

    acceptable:false

    IMPORTANT: 
    - Do ALL your thinking internally
    - Output ONLY the exact text above
    - No explanations, no reasoning, no additional words
    - Just the exact format: acceptable:true or acceptable:false
    """
    
    result = call_llm(state, llm, ranker_prompt)
    # Store the ranker evaluation in state
    if result["messages"] and hasattr(result["messages"][-1], 'content'):
        result["ranker_evaluation"] = result["messages"][-1].content
    return result


def pr_processing_llm_bound(state, llm, config):
    """
    PR agent that processes the final answer with proper context.
    """
    print("🎯 PR Agent: Processing final answer")
    
    # Use the stored values from state instead of extracting from messages
    original_question = state.get("original_question", "Unknown")
    retrieved_content = state.get("retrieved_content", "")
    ranker_evaluation = state.get("ranker_evaluation", "")
    retrieval_attempts = state.get("retrieval_attempts", 0)
    
    print(f"📝 PR Agent Context:")
    print(f"  - Original Question: {original_question}")
    print(f"  - Retrieved Content Length: {len(retrieved_content) if retrieved_content else 0}")
    print(f"  - Ranker Evaluation: {ranker_evaluation}")
    print(f"  - Retrieval Attempts: {retrieval_attempts}")
    
    # Check if this is a direct answer from Assistant (no retrieval was performed)
    is_direct_answer = (retrieval_attempts == 0 and 
                       not retrieved_content and
                       not ranker_evaluation)
    
    if is_direct_answer:
        print("🔄 Processing direct answer from Assistant (no retrieval performed)")
        
        # Get the Assistant's existing answer from messages
        assistant_answer = ""
        if state["messages"] and len(state["messages"]) > 0:
            # Find the Assistant's response (should be the last message)
            last_message = state["messages"][-1]
            if hasattr(last_message, 'content'):
                assistant_answer = last_message.content
                print(f"📄 Assistant's existing answer: {assistant_answer[:200]}...")
        
        # For direct answers, we need to create a modified state that includes the Assistant's answer
        # as the context for polishing, rather than the original user message
        polishing_state = {
            "messages": [
                {"role": "user", "content": f"Original question: {original_question}\nAssistant's answer to polish: {assistant_answer}"}
            ],
            "original_question": original_question,
            "retrieved_content": retrieved_content,
            "ranker_evaluation": ranker_evaluation,
            "retrieval_attempts": retrieval_attempts
        }
        
        # For direct answers, we still want PR polishing but with the existing answer as context
        direct_answer_prompt = f"""
        You are a Public Relations and Finalization Agent using ReAct (Reasoning and Acting) methodology.

        TASK: Take the Assistant's existing answer and polish it for clarity, professionalism, and user-friendliness. 
        PRESERVE the core meaning and information while improving the presentation.

        ASSISTANT'S ORIGINAL ANSWER TO POLISH: {assistant_answer}

        USER'S ORIGINAL QUESTION: {original_question}

        IMPORTANT INSTRUCTIONS:
        - Your ONLY task is to POLISH the existing answer, not create a new one
        - PRESERVE all key information and meaning from the Assistant's answer
        - Improve clarity, formatting, grammar, and professional tone
        - Answer in the SAME LANGUAGE as the original question
        - Use proper paragraphs, bullet points, or other formatting as appropriate
        - DO NOT add new information or change the fundamental meaning
        - DO NOT ask follow-up questions or change the response type

        Your output should be the POLISHED VERSION of the Assistant's answer, ready to be shown to the user.
        """
        
        return call_llm(polishing_state, llm, direct_answer_prompt)
    
    # ... rest of existing PR agent logic for retrieval-based answers ...
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
    You are a Public Relations and Finalization Agent using ReAct (Reasoning and Acting) methodology.

    USER'S ORIGINAL QUESTION: {original_question}
    RETRIEVED INFORMATION: {retrieved_content}
    RANKER EVALUATION: {ranker_evaluation}
    RETRIEVAL ATTEMPTS: {retrieval_attempts} / 5
    {max_attempts_context}

    TASK: Create a polished, comprehensive final answer for the user.

    Use this internal ReAct reasoning process (do not output this thinking):

    THOUGHT: I need to synthesize all available information into a clear, helpful response that directly addresses the user's question in the appropriate language.

    REASONING: Analyze the complete context systematically:

    1. LANGUAGE IDENTIFICATION: Determine the language of the original question to ensure response alignment
    2. CONTENT ASSESSMENT: Evaluate the quality and quantity of retrieved information and determine what can be confidently answered
    3. LANGUAGE PROCESSING: If retrieved content is in a different language than the question, prepare translations while maintaining accuracy and context
    4. ANSWER STRATEGY: Choose appropriate response approach based on information availability - comprehensive answer, partial answer with limitations, or polite acknowledgment of insufficient information
    5. FORMATTING STRATEGY: Structure content for maximum clarity using appropriate formatting, proper paragraphs, bullet points, and professional tone

    ACTION: Synthesize information into final response following optimal strategy, language alignment, and formatting approach.

    OBSERVATION: Review the planned response for completeness, accuracy, language consistency, and user-friendliness.

    CONCLUSION: Deliver the final polished answer that best serves the user's needs given available information.

    IMPORTANT: Do ALL your thinking internally. Your output should be the FINAL ANSWER that will be shown directly to the user. 

    CRITICAL LANGUAGE REQUIREMENTS:
    - Answer in the SAME LANGUAGE as the original question
    - If the original question is in English, provide your answer in English
    - If retrieved content is in a different language than the question, provide a translation in the question's language
    - For non-English questions with non-English content: provide the answer in the question's language
    - For English questions with non-English content: provide English translation and mention the original language if relevant

    Create a response that:
    - Directly addresses the user's question in the appropriate language
    - Uses clear, professional language
    - Includes proper formatting for readability
    - Provides translations when content language differs from question language
    - Acknowledges limitations honestly when necessary
    - Maintains a helpful and accessible tone

    If insufficient information is available, provide a polite response in the question's language explaining the limitation while offering what assistance you can.
    """
    
    return call_llm(state, llm, comprehensive_prompt)