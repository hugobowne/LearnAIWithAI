import os
import openai
import argparse
import re # Import regex for VTT parsing
from dotenv import load_dotenv

def load_vtt_content(file_path):
    """Reads a VTT file and extracts the text content, skipping metadata and timestamps."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading transcript file: {e}")
        return None

    content_lines = []
    is_content = False
    for line in lines:
        line = line.strip()
        # Skip empty lines, WEBVTT header, and timestamp lines
        if not line or line == 'WEBVTT' or '-->' in line:
            is_content = False
            continue
        # Skip lines that look like metadata (e.g., NOTE, STYLE)
        if re.match(r'^[A-Z]+(\s*:.*)?$', line):
             is_content = False
             continue
        # If it's not metadata or timestamp, assume it's content
        # A simple heuristic: content often follows a timestamp line
        # A better check might be needed for complex VTTs
        # We will just append any line that doesn't match the skip conditions
        content_lines.append(line)
        
    return " ".join(content_lines)

def answer_question_basic(client, context, question):
    """Minimal function to ask OpenAI a question based on provided context."""
    system_prompt = """
    You are a helpful assistant. Answer questions based ONLY on the provided context from the workshop transcript.
    If the answer is not in the context, say you don't know based on the provided transcript.
    Keep your answers concise.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k", # Use 16k model for potentially long transcripts
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context (Workshop Transcript):\n\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.1, 
            max_tokens=500 # Allow more tokens for answers from transcript
        )
        return response.choices[0].message.content
    except Exception as e:
        # Add more specific error handling if needed (e.g., context length) 
        return f"An error occurred interacting with OpenAI: {str(e)}"

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Answer a question based on hardcoded context using OpenAI.")
    parser.add_argument("question", type=str, help="The question to ask based on the context.")
    args = parser.parse_args()
    question = args.question
    # --- End Argument Parsing ---

    # Load environment variables (expects OPENAI_API_KEY in .env file)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        print("Please ensure you have a .env file in the project root with OPENAI_API_KEY=your-key")
        return
        
    # Initialize the OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    # --- Determine and Load Transcript File --- 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path relative to the script location
    transcript_file = os.path.join(script_dir, "../data/WS1-C2.vtt") 
    print(f"Loading transcript from: {transcript_file}...")
    workshop_context = load_vtt_content(transcript_file)

    if workshop_context is None:
        return # Exit if loading failed
    print("Transcript loaded successfully.")
    
    # --- Truncate context if too long for the model --- #
    MAX_CHARS = 60000 # Approx 15k tokens (using 4 chars/token heuristic)
    original_length = len(workshop_context)
    if original_length > MAX_CHARS:
        print(f"\n[Warning] Transcript is too long ({original_length} chars). Truncating to first {MAX_CHARS} characters.")
        print("Answers will be based only on the beginning of the transcript.")
        workshop_context = workshop_context[:MAX_CHARS]
    # --- End Truncation --- #
    
    # --- Transcript Loading Complete ---

    print("\nBasic Q&A System (Command-Line Input)")
    print("---------------------------------------")
    # print("Using the following context:") # Optional: Might be too verbose for CLI
    # print(workshop_context)
    print(f"Question: {question}")
    
    # Call the QA function with the provided question
    answer = answer_question_basic(client, workshop_context, question)
    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main() 