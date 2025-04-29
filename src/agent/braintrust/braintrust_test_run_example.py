import requests
import os
from dotenv import load_dotenv
import json
import time # Import time for potential delays

# Load environment variables
load_dotenv()

API_KEY = os.getenv("BRAINTRUST_API_KEY")
PROJECT_ID = "7b67d69f-2f0d-4102-a5fd-e448681d6627"

if not API_KEY:
    print("Error: BRAINTRUST_API_KEY not found in environment variables.")
    exit(1)

print(f"Attempting to fetch ALL logs via paginated BTQL API for project ID: {PROJECT_ID}")

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

API_BASE_URL = "https://api.braintrust.dev/"
BTQL_ENDPOINT = f"{API_BASE_URL}btql"

all_logs = []
cursor = None
page_num = 1
BATCH_SIZE = 100 # Fetch 100 logs per request

while True:
    print(f"\nFetching page {page_num} (limit={BATCH_SIZE}, cursor={cursor})...")
    # Construct the BTQL query using correct syntax, without a limit
    btql_query = f"""
    select: *
    from: project_logs('{PROJECT_ID}')
    """
    if cursor:
        # Note: Docs aren't explicit on cursor *syntax* in BTQL, assuming it's a top-level param?
        # Might need adjustment if this assumption is wrong.
        # UPDATE: Example shows it in the request body, not the query string itself.
        pass # Cursor will be added to request_body below

    request_body = {
        "query": btql_query,
        "fmt": "json",
        "cursor": cursor # Add cursor to the request body
    }

    try:
        response = requests.post(BTQL_ENDPOINT, headers=headers, json=request_body)
        response.raise_for_status()

        response_data = response.json()
        current_batch_logs = response_data.get('data', [])
        print(f"Fetched {len(current_batch_logs)} logs this page.")

        if not current_batch_logs:
            print("No more logs found.")
            break # Exit loop if no logs in this batch

        all_logs.extend(current_batch_logs)
        print(f"Total logs fetched so far: {len(all_logs)}")

        # Check for the next cursor in headers
        next_cursor = response.headers.get("x-bt-cursor") or response.headers.get("x-amz-meta-bt-cursor")
        
        if next_cursor and next_cursor != cursor: # Ensure cursor has changed
            cursor = next_cursor
            page_num += 1
            # Optional: add a small delay between requests
            # time.sleep(0.1)
        else:
            print("No next cursor found or cursor did not change, stopping pagination.")
            break # Exit loop if no next cursor

    except requests.exceptions.RequestException as req_err:
        print(f"\nHTTP Request Error on page {page_num}: {req_err}")
        if hasattr(req_err, 'response') and req_err.response is not None:
            print(f"Status Code: {req_err.response.status_code}")
            try:
                print(f"Response Body: {req_err.response.json()}")
            except json.JSONDecodeError:
                print(f"Response Body: {req_err.response.text}")
        print("Stopping pagination due to error.")
        break
    except Exception as e:
        print(f"\nAn unexpected error occurred on page {page_num}: {e}")
        print("Stopping pagination due to error.")
        break

# --- Processing fetched logs ---

print(f"\nFinished fetching. Total logs retrieved: {len(all_logs)}")

if all_logs:
    # Save all logs to a file
    output_filename = "fetched_braintrust_logs.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_logs, f, indent=2)
        print(f"Successfully saved {len(all_logs)} logs to {output_filename}")
        
        # Search for the hand label within the fetched data
        found_label = False
        print(f"\nSearching for 'Final Answer Quality' score in {len(all_logs)} logs...")
        for i, log in enumerate(all_logs):
            scores = log.get('scores')
            if scores and isinstance(scores, dict) and "Final Answer Quality" in scores:
                print(f"Found 'Final Answer Quality' in log record {i+1} (ID: {log.get('id')})")
                # Optionally print the specific log here
                # print(json.dumps(log, indent=2))
                found_label = True
                # break # Uncomment to stop after finding the first instance
        
        if not found_label:
            print("'Final Answer Quality' score not found in the fetched logs.")
            
    except Exception as write_err:
        print(f"Error saving logs to file: {write_err}")

else:
    print("No logs were fetched.") 