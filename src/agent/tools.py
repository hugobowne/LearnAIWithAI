import sqlite3
import json
from pathlib import Path
from opentelemetry import trace # Import trace
from opentelemetry.trace import Status, StatusCode # Import Status codes

# --- Constants ---
# Assuming the script using this module is run from the project root
DB_FILE_PATH = Path("data/workshop1_transcript.db")
TABLE_NAME = "transcript_segments"

# Get a tracer instance for this module
tracer = trace.get_tracer("agent.tools")

# --- Tool Functions ---

@tracer.start_as_current_span("query_database_tool", kind=trace.SpanKind.INTERNAL)
def query_database(sql_query: str) -> str:
    """Executes a SQL query against the transcript database and returns the results."""
    span = trace.get_current_span()
    span.set_attribute("db.system", "sqlite")
    span.set_attribute("db.statement", sql_query)
    span.set_attribute("db.table", TABLE_NAME)
    
    print(f"--- Executing Tool: query_database --- ")
    print(f"SQL Query: {sql_query}")

    if not DB_FILE_PATH.exists():
        error_msg = f"Error: Database file not found at {DB_FILE_PATH}"
        span.record_exception(FileNotFoundError(error_msg))
        span.set_status(Status(StatusCode.ERROR, error_msg))
        return error_msg

    conn = None
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        # Set row_factory to return dictionaries for easier handling
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Convert results (list of sqlite3.Row) to a list of dictionaries
        results_list = [dict(row) for row in results]

        if not results_list:
            result_str = "No matching records found in the database for this query."
            print("Result: No matching records found.")
            span.set_attribute("db.results.count", 0)
        else:
            results_json = json.dumps(results_list)
            result_str = results_json
            span.set_attribute("db.results.count", len(results_list))
            # Log truncated result to avoid overly large spans
            span.set_attribute("db.results.preview", results_json[:500] + ("..." if len(results_json) > 500 else ""))
            print(f"Result (first 500 chars): {results_json[:500]}...")
            
        span.set_status(Status(StatusCode.OK))
        return result_str

    except sqlite3.Error as e:
        error_msg = f"Error executing SQL query: {e}. Please check the query syntax and table/column names."
        print(f"SQLite Error: {e}")
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, f"SQLite Error: {e}"))
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while querying the database: {e}"
        print(f"Unexpected Error: {e}")
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, f"Unexpected Error: {e}"))
        return error_msg
    finally:
        if conn:
            conn.close()
        print("--- Tool Execution Finished --- ")

# --- Tool Schema Definitions ---

def get_tool_schemas():
    """Returns a list containing the JSON schemas for all available tools."""
    
    # Schema for the query_database tool
    query_db_schema = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Executes a read-only SQL query against a database containing transcript segments from Workshop 1. Use this to find specific information mentioned in the workshop transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": f"A valid SQLite SELECT query targeting the '{TABLE_NAME}' table. The table schema is: (segment_id INTEGER PRIMARY KEY, session_name TEXT, start_time_seconds REAL, end_time_seconds REAL, speaker TEXT, text TEXT, word_count INTEGER). Filter based on user request (e.g., time, speaker, keywords in text). Query only the columns you need."
                    }
                },
                "required": ["sql_query"]
            }
        }
    }

    # In the future, add schemas for other tools (analyze_data, visualize_data) here
    # analyze_data_schema = { ... }
    # visualize_data_schema = { ... }

    # Return a list containing all tool schemas
    return [query_db_schema]

# --- Example Usage (for testing the tool directly) ---
if __name__ == '__main__':
    print("Testing query_database tool...")
    
    # Example 1: Find segments by speaker
    test_query_1 = "SELECT segment_id, text FROM transcript_segments WHERE speaker = 'hugo bowne-anderson' LIMIT 3"
    print(f"\nRunning Test Query 1: {test_query_1}")
    result_1 = query_database(test_query_1)
    print(f"Result 1:\n{result_1}")

    # Example 2: Find segments mentioning 'evaluation'
    test_query_2 = "SELECT segment_id, speaker, text FROM transcript_segments WHERE text LIKE '%evaluation%' LIMIT 3"
    print(f"\nRunning Test Query 2: {test_query_2}")
    result_2 = query_database(test_query_2)
    print(f"Result 2:\n{result_2}")

    # Example 3: Invalid query (syntax error)
    test_query_3 = "SELECT * FRO transcript_segments LIMIT 1"
    print(f"\nRunning Test Query 3 (Invalid): {test_query_3}")
    result_3 = query_database(test_query_3)
    print(f"Result 3:\n{result_3}")

    # Example 4: Query for non-existent data
    test_query_4 = "SELECT * FROM transcript_segments WHERE speaker = 'NonExistentSpeaker'"
    print(f"\nRunning Test Query 4 (No Results): {test_query_4}")
    result_4 = query_database(test_query_4)
    print(f"Result 4:\n{result_4}") 