import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path # Import Path

# Tracing Imports
import phoenix as px # Alias phoenix
from phoenix.otel import register # <-- Use register function from notebook
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Import tool functions (changed to direct import for simplicity)
from tools import query_database, get_tool_schemas

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Consider adding more robust error handling or instructions
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please ensure it's set in your .env file.")

client = OpenAI(api_key=api_key)
MODEL = "gpt-4o" # Use the full gpt-4o model

# --- Phoenix/OpenTelemetry Tracing Setup ---
PROJECT_NAME = "transcript-agent-mvp"
try:
    # --- Explicit Configuration - Attempt 2 --- 
    # Bypass environment variable detection within register().
    # Define endpoint explicitly with the correct path.
    endpoint = "https://app.phoenix.arize.com/v1/traces"
    
    # Get the required header string from the env var specified in docs.
    headers_str = os.getenv("PHOENIX_CLIENT_HEADERS") # Use PHOENIX_ var

    if not headers_str:
        # Adjusted error message to reflect the correct variable name
        raise ValueError("PHOENIX_CLIENT_HEADERS not found in environment variables.")

    # Parse the header string (e.g., "api_key=value") into a dictionary.
    headers_dict = {}
    try:
        # Simple parsing for "key1=value1,key2=value2,..."
        for item in headers_str.split(','):
            key, value = item.split('=', 1)
            headers_dict[key.strip()] = value.strip()
        if not headers_dict: # Raise error if parsing resulted in empty dict
             raise ValueError("Parsed headers dictionary is empty.")
    except Exception as parse_err:
        # Adjusted error message
        print(f"⚠️ ERROR: Could not parse PHOENIX_CLIENT_HEADERS string: '{headers_str}'. Error: {parse_err}")
        raise ValueError("Invalid PHOENIX_CLIENT_HEADERS format") from parse_err

    print(f"Attempting explicit Phoenix tracing configuration.")
    print(f"  Endpoint: {endpoint}")
    print(f"  Headers: { {k: '****' for k in headers_dict} }") # Print header keys safely

    phoenix_tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint=endpoint,        # Pass explicit endpoint
        headers=headers_dict        # Pass explicit, parsed headers dictionary
        # No protocol needed, inferred from https:// in endpoint
    )
    OpenAIInstrumentor().instrument(tracer_provider=phoenix_tracer_provider)
    tracer = trace.get_tracer("agent.agent") 
    print(f"✅ Phoenix tracing explicitly configured for project '{PROJECT_NAME}'.")
except ValueError as ve:
    print(f"⚠️ Configuration Error for Phoenix tracing: {ve}. Tracing disabled.")
    tracer = None
except Exception as e:
    print(f"⚠️ Failed to initialize Phoenix tracing: {e}. Tracing disabled.")
    tracer = None # Set tracer to None if setup fails
# --- End Tracing Setup ---

# Store available tools {tool_name: function}
def get_available_tools():
    return {
        "query_database": query_database,
        # Add other tools here later
    }

# Main agent function - Now returns a detailed dictionary
# Add span decorator to trace the whole conversation function
@tracer.start_as_current_span("run_agent_conversation", kind=trace.SpanKind.SERVER)
def run_agent_conversation(user_query: str) -> dict:
    """Handles a single turn of conversation and returns detailed execution log."""
    span = trace.get_current_span() # Get current span
    span.set_attribute("input.value", user_query)
    print(f"\n{'='*20} New Query {'='*20}")
    print(f"User Query: {user_query}")
    
    run_log = {
        "user_query": user_query,
        "system_prompt": "",
        "messages": [],
        "final_answer": None,
        "error": None
    }
    
    # --- Agent Logic --- 
    try: # Wrap main logic in try/except to ensure span status is set
        # 1. Define System Prompt
        system_prompt = (f"""
        You are a helpful assistant designed to answer questions about the LearnAIWithAI Workshop 1 transcript. 
        Use the available tools to query the transcript database when necessary. 
        The database table is '{get_tool_schemas()[0]['function']['parameters']['properties']['sql_query']['description'].split('targeting the ')[1].split(' table.')[0]}' and contains segments of the transcript.
        Base your answers SOLELY on the information retrieved from the database using the tools. 
        If the information is not found in the database, say that you cannot answer the question based on the available transcript data.
        Be concise and directly answer the user's query based on the tool results.
        """)
        run_log["system_prompt"] = system_prompt
        span.set_attribute("llm.system_prompt", system_prompt) # Add system prompt to span

        # 2. Define Tools
        available_tools = get_available_tools()
        tool_schemas = get_tool_schemas()

        # 3. Manage Conversation History 
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        run_log["messages"].append(messages[-1]) # Log user message

        print("\n--- Sending request to LLM (Attempt 1) ---")
        # 4. Call OpenAI API (First Attempt) - Auto-instrumented by OpenAIInstrumentor
        # We can add a span here if we want to group this specific call
        with tracer.start_as_current_span("llm_call_1") as llm_span_1:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto"
            )
            response_message = response.choices[0].message
        
        messages.append(response_message) 
        run_log["messages"].append(response_message.model_dump()) 

        # 5. Process Response
        tool_calls = response_message.tool_calls

        if tool_calls:
            print("--- LLM requested tool call(s) ---")
            # Add a span around the tool execution loop
            with tracer.start_as_current_span("tool_execution_loop") as tool_loop_span:
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_tools.get(function_name)
                    tool_error = None
                    function_response = None
                    tool_loop_span.set_attribute(f"tool.{function_name}.id", tool_call.id)
                    
                    if not function_to_call:
                        tool_error = f"Error: Tool '{function_name}' not found."
                        print(tool_error)
                        function_response = tool_error 
                        tool_loop_span.set_status(Status(StatusCode.ERROR, tool_error))
                    else:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            tool_loop_span.set_attribute(f"tool.{function_name}.args", json.dumps(function_args)) # Log args
                            print(f"Calling tool: {function_name} with args: {function_args}")
                            # The actual tool call is traced by the decorator in tools.py
                            function_response = function_to_call(**function_args)
                            tool_loop_span.set_attribute(f"tool.{function_name}.response_preview", str(function_response)[:500]) # Log response preview
                        except json.JSONDecodeError as e:
                            tool_error = f"Error: Invalid arguments format for tool '{function_name}'. Expected JSON."
                            print(f"{tool_error} Got: {tool_call.function.arguments}")
                            function_response = tool_error
                            tool_loop_span.record_exception(e)
                            tool_loop_span.set_status(Status(StatusCode.ERROR, tool_error))
                        except Exception as e:
                            tool_error = f"Error executing tool '{function_name}': {e}"
                            print(tool_error)
                            function_response = tool_error
                            tool_loop_span.record_exception(e)
                            tool_loop_span.set_status(Status(StatusCode.ERROR, tool_error))
                    
                    # Append the tool's response to the message history
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response
                    })
                    run_log["messages"].append(messages[-1]) # Log tool message

            # 6. Call OpenAI API again with tool results - Auto-instrumented
            print("\n--- Sending request to LLM (Attempt 2, with tool results) ---")
            with tracer.start_as_current_span("llm_call_2") as llm_span_2:
                second_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages 
                )
                final_response_message = second_response.choices[0].message
            
            messages.append(final_response_message)
            run_log["messages"].append(final_response_message.model_dump()) # Log LLM response 2
            span.set_attribute("output.value", final_response_message.content)
            run_log["final_answer"] = final_response_message.content
            print("--- Received final response from LLM ---")
        
        else:
            # No tool call was made
            print("--- LLM did not request tool call, returning response directly ---")
            span.set_attribute("output.value", response_message.content)
            run_log["final_answer"] = response_message.content
            
        span.set_status(Status(StatusCode.OK))
        return run_log
    
    except Exception as e:
        # Record exception and set error status on the main span
        print(f"Error during agent conversation: {e}")
        run_log["error"] = f"Error during agent conversation: {e}"
        if span:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
        return run_log

# Example Usage Block (Writes results to JSON log file)
if __name__ == "__main__":
    # Remove the outer span wrapper - each run_agent_conversation call
    # will now be its own root trace thanks to its decorator.
    print("Running agent with predefined questions from JSON file...")
    
    queries_file_path = Path(__file__).parent / "docs" / "test_queries.json"
    log_file_path = Path(__file__).parent / "agent_run_log.json" 
    all_run_logs = [] 
    
    if not queries_file_path.exists():
        print(f"Error: Test queries file not found at {queries_file_path}")
        exit(1)

    try:
        with open(queries_file_path, 'r') as f:
            example_queries = json.load(f)
        if not isinstance(example_queries, list):
             raise ValueError("Test queries file should contain a JSON list of strings.")
             
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing {queries_file_path}: {e}")
        exit(1)

    print(f"Loaded {len(example_queries)} queries from {queries_file_path}")

    # Loop through queries directly
    for query in example_queries:
        try:
            run_log_result = run_agent_conversation(query)
            all_run_logs.append(run_log_result)
            print(f"\nAgent Response: {run_log_result.get('final_answer', '[No answer generated]')}\n") 
            print(f"{'-'*50}\n") # Separator
        except Exception as e:
            print(f"\nCritical Error processing query '{query}': {e}\n")
            all_run_logs.append({"user_query": query, "error": f"Critical Error: {e}"})
            print(f"{'-'*50}\n") # Separator

    # (Log writing logic remains the same)
    print(f"\nWriting detailed logs to {log_file_path}...")
    try:
        with open(log_file_path, 'w') as f:
            json.dump(all_run_logs, f, indent=4) 
        print("Logs written successfully.")
    except Exception as e:
        print(f"Error writing log file: {e}")

    print("Finished processing predefined questions.") 