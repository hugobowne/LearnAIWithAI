import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path # Import Path

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
MODEL = "gpt-4o-mini" # Or "gpt-4o" if needed later

# Store available tools {tool_name: function}
def get_available_tools():
    return {
        "query_database": query_database,
        # Add other tools here later
    }

# Main agent function - Now returns a detailed dictionary
def run_agent_conversation(user_query: str) -> dict:
    """Handles a single turn of conversation and returns detailed execution log."""
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
    # 4. Call OpenAI API (First Attempt)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        messages.append(response_message) 
        run_log["messages"].append(response_message.model_dump()) # Log LLM response 1

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        run_log["error"] = f"Error calling OpenAI API (Attempt 1): {e}"
        return run_log # Return early on error

    # 5. Process Response
    tool_calls = response_message.tool_calls

    if tool_calls:
        print("--- LLM requested tool call(s) ---")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_tools.get(function_name)
            tool_error = None
            function_response = None
            
            if not function_to_call:
                 tool_error = f"Error: Tool '{function_name}' not found."
                 print(tool_error)
                 function_response = tool_error 
            else:
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"Calling tool: {function_name} with args: {function_args}")
                    function_response = function_to_call(**function_args)
                except json.JSONDecodeError:
                    tool_error = f"Error: Invalid arguments format for tool '{function_name}'. Expected JSON."
                    print(f"{tool_error} Got: {tool_call.function.arguments}")
                    function_response = tool_error
                except Exception as e:
                    tool_error = f"Error executing tool '{function_name}': {e}"
                    print(tool_error)
                    function_response = tool_error
            
            # Append the tool's response to the message history
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })
            run_log["messages"].append(messages[-1]) # Log tool message

        # 6. Call OpenAI API again with tool results
        print("\n--- Sending request to LLM (Attempt 2, with tool results) ---")
        try:
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages 
            )
            final_response_message = second_response.choices[0].message
            messages.append(final_response_message)
            run_log["messages"].append(final_response_message.model_dump()) # Log LLM response 2
            run_log["final_answer"] = final_response_message.content
            print("--- Received final response from LLM ---")
        except Exception as e:
            print(f"Error calling OpenAI API on second attempt: {e}")
            run_log["error"] = f"Error calling OpenAI API (Attempt 2): {e}"
            
        return run_log
    
    else:
        # No tool call was made
        print("--- LLM did not request tool call, returning response directly ---")
        run_log["final_answer"] = response_message.content
        return run_log

# Example Usage Block (Writes results to JSON log file)
if __name__ == "__main__":
    print("Running agent with predefined questions from JSON file...")

    queries_file_path = Path(__file__).parent / "docs" / "test_queries.json"
    # Log file inside the agent directory
    log_file_path = Path(__file__).parent / "agent_run_log.json" 
    all_run_logs = [] # List to store logs for all queries
    
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

    for query in example_queries:
        try:
            run_log_result = run_agent_conversation(query)
            all_run_logs.append(run_log_result)
            # Print only the final answer to keep console output cleaner
            print(f"\nAgent Response: {run_log_result.get('final_answer', '[No answer generated]')}\n") 
            print(f"{'-'*50}\n") # Separator
        except Exception as e:
            print(f"\nCritical Error processing query '{query}': {e}\n")
            # Log the error case as well
            all_run_logs.append({"user_query": query, "error": f"Critical Error: {e}"})
            print(f"{'-'*50}\n") # Separator

    # Write all logs to the JSON file
    print(f"\nWriting detailed logs to {log_file_path}...")
    try:
        with open(log_file_path, 'w') as f:
            json.dump(all_run_logs, f, indent=4) # Use indent for readability
        print("Logs written successfully.")
    except Exception as e:
        print(f"Error writing log file: {e}")

    print("Finished processing predefined questions.") 