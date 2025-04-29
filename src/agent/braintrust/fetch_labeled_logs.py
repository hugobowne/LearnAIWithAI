import braintrust
import os
from dotenv import load_dotenv
import json

# Load environment variables (for BRAINTRUST_API_KEY)
load_dotenv()

# Project ID (Required for fetching logs via the projects API)
PROJECT_ID = "7b67d69f-2f0d-4102-a5fd-e448681d6627"

print(f"Fetching logs via braintrust.projects.logs.fetch for project ID: {PROJECT_ID}...")

try:
    # Directly call the documented function, assuming API key is handled via environment
    # Omit the filter query for now to fetch all logs (up to default limit)
    fetched_logs = braintrust.projects.logs.fetch(project_id=PROJECT_ID, limit=10)

    # Check if 'events' key exists and has items
    if fetched_logs and 'events' in fetched_logs and fetched_logs['events']:
        print(f"\n--- Fetched {len(fetched_logs['events'])} log records ---")
        
        # Print the first record
        first_record = fetched_logs['events'][0]
        print("\n--- Raw data for the first fetched log record ---")
        print(json.dumps(first_record, indent=2))
            
    else:
        print("No log records found or the response format is unexpected.")
        print("Raw response:", fetched_logs) # Print raw response for debugging

except AttributeError as ae:
    print(f"An AttributeError occurred: {ae}")
    print("This might indicate the structure 'braintrust.projects.logs.fetch' is incorrect or requires a different client/initialization.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 