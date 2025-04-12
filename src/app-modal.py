import modal
from pathlib import Path # Add pathlib import

# Create a container image with necessary dependencies
image = (
    modal.Image.debian_slim()
    .pip_install("gradio", "fastapi", "openai>=1.1.0", "chromadb>=0.4.18", "tiktoken>=0.5.0", "numpy>=1.24.0", "python-dotenv>=1.0.0", "pysqlite3-binary")
    # Use path relative to project root for mounting
    .add_local_dir("data", remote_path="/data", copy=True)
)

# Define the Modal app
app = modal.App("rag-app", image=image)

# Define persistent volume for logs
logs_db_storage = modal.Volume.from_name("rag-app-logs", create_if_missing=True)

# Modal function to launch the Gradio app
@app.function(
    max_containers=1, # Limit concurrency as SQLite isn't designed for concurrent writes
    allow_concurrent_inputs=1000, # Allow many inputs to the single container
    secrets=[modal.Secret.from_name("openai-secret")],
    volumes={"/logs_db": logs_db_storage} # Mount volume
)
@modal.asgi_app()
def run_gradio():
    # Workaround for pysqlite3 on Modal
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

    import asyncio # Add asyncio import
    from contextlib import asynccontextmanager # Add asynccontextmanager import
    import time
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    # Update imports for relative execution
    from .simple_rag import (
        answer_question, process_workshop, init_logs_db, WORKSHOP_TRANSCRIPT_PATH, 
        COLLECTION_NAME, log_interaction, COMPLETION_MODEL, LOGS_DB_PATH, 
        log_feedback # Add feedback function import
    )
    import sqlite3 # Add sqlite3 import
    from fastapi.responses import JSONResponse # Add JSONResponse import

    # --- Persistence Logic ---
    remote_db_path = Path("/logs_db") / LOGS_DB_PATH
    local_db_path = Path(".") / LOGS_DB_PATH
    if remote_db_path.exists():
        local_db_path.write_bytes(remote_db_path.read_bytes())
        print(f"Copied existing database from volume: {remote_db_path}")
    else:
        print("No existing database found on volume. Initializing.")

    def persist():
        if local_db_path.exists():
            print(f"Persisting {local_db_path} to {remote_db_path}")
            remote_db_path.write_bytes(local_db_path.read_bytes())
            logs_db_storage.commit()
        else:
            print(f"Local db file {local_db_path} not found for persistence.")

    async def persist_background():
        while True:
            await asyncio.sleep(60) # Persist every 60 seconds
            persist()

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        # Start background persistence on startup
        background_task = asyncio.create_task(persist_background())
        print("Background persistence started.")
        yield
        # Persist on shutdown
        print("Persisting database on shutdown...")
        persist()
        background_task.cancel() # Clean up background task
        print("Shutdown persistence complete.")
    # --- End Persistence Logic ---


    # Initialize workshop content & DB (will create local file if not exists)
    print("Initializing logs database...")
    init_logs_db()
    print("Processing workshop transcript...")
    process_workshop(WORKSHOP_TRANSCRIPT_PATH, COLLECTION_NAME)
    print("Initialization complete.")

    # Pass lifespan manager to FastAPI
    api = FastAPI(lifespan=lifespan)

    # --- Add /logs endpoint --- 
    @api.get("/logs")
    def get_logs():
        print(f"Accessing /logs endpoint. Checking for {local_db_path}")
        if not local_db_path.exists():
            print("Logs database file not found.")
            return JSONResponse(content={"error": "Logs database not found."}, status_code=404)
        
        try:
            conn = sqlite3.connect(local_db_path)
            conn.row_factory = sqlite3.Row # Return rows as dict-like objects
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, question, response, latency, num_chunks, total_tokens, "
                "feedback_rating, feedback_reason, feedback_notes, feedback_user " # Added feedback columns
                "FROM logs ORDER BY timestamp DESC LIMIT 10"
            )
            logs = [dict(row) for row in cursor.fetchall()]
            conn.close()
            print(f"Returning {len(logs)} log entries.")
            return JSONResponse(content=logs)
        except Exception as e:
            print(f"Error reading logs database: {e}")
            return JSONResponse(content={"error": f"Failed to read logs: {e}"}, status_code=500)
    # --- End /logs endpoint ---

    def answer(query):
        start_time = time.time()
        print(f"Received query: {query}")
        response_text, _, context_info = answer_question(query)
        end_time = time.time()
        log_id = log_interaction(query, response_text, context_info, COMPLETION_MODEL, start_time, end_time)
        print(f"Generated response in {end_time - start_time:.2f} seconds. Log ID: {log_id}")
        
        # Return response, log_id, make feedback visible and clear fields, clear status
        return (
            response_text, 
            log_id, 
            gr.update(visible=True), # Make group visible
            gr.update(value=None), # Clear rating
            gr.update(value=""),   # Clear reason
            gr.update(value=""),   # Clear notes
            gr.update(value=""),   # Clear user
            ""                     # Clear status message
        )

    # Wrapper function to handle feedback submission and UI updates
    def submit_feedback_and_update_ui(log_id, rating, reason, notes, user):
        if not reason or not reason.strip():
            # Keep feedback form visible, show error
            return gr.update(value="Reason is required.", visible=True), gr.update(visible=True)
        
        success = log_feedback(log_id, rating, reason, notes, user)
        
        if success:
            # Hide feedback form, show success message
            return gr.update(value="Feedback submitted. Thank you!", visible=True), gr.update(visible=False)
        else:
            # Keep feedback form visible, show error message
            return gr.update(value="Error submitting feedback. Please check logs.", visible=True), gr.update(visible=True)

    # Gradio app interface setup
    with gr.Blocks() as blocks:
        gr.Markdown("## LearnWithAI RAG App (v4 - Persistent Logs + Feedback)") # Update title
        current_log_id = gr.State(None) # State to hold the ID of the interaction being reviewed

        # Place components vertically
        query_input = gr.Textbox(label="Ask a question", show_label=False, placeholder="Ask a question about the workshop...") # Use placeholder, hide label for cleaner look
        query_button = gr.Button("Submit")
        
        output = gr.Textbox(label="Answer", interactive=False)
        
        # --- Feedback Components (Initially Hidden) ---
        with gr.Group(visible=False) as feedback_group:
            gr.Markdown("### Rate this Response")
            feedback_rating = gr.Radio(["PASS", "FAIL"], label="Rating")
            feedback_reason = gr.Textbox(label="Reason for Rating (Required)", placeholder="Provide justification...")
            feedback_notes = gr.Textbox(label="Additional Notes (Optional)", placeholder="Latency issues, other comments...")
            feedback_user = gr.Textbox(label="Your Name (Optional)")
            feedback_button = gr.Button("Submit Feedback")
            feedback_status = gr.Markdown("") # To show submission status
            
        # --- Gradio Actions ---
        
        # When query button is clicked
        query_button.click(
            answer, 
            inputs=[query_input], 
            # Update outputs list to match the number of return values from answer
            outputs=[
                output, 
                current_log_id, 
                feedback_group, 
                feedback_rating, # Add components to be cleared
                feedback_reason,
                feedback_notes,
                feedback_user,
                feedback_status
            ]
        )
        
        # When feedback button is clicked
        feedback_button.click(
            submit_feedback_and_update_ui, # Call the wrapper function
            inputs=[
                current_log_id, 
                feedback_rating, 
                feedback_reason, 
                feedback_notes, 
                feedback_user
            ],
            outputs=[feedback_status, feedback_group] # Update feedback group visibility on submit
        )

        # Mount Gradio within FastAPI
        return mount_gradio_app(app=api, blocks=blocks, path="/")