"""
Simple RAG Implementation for Workshop Q&A

A simplified version of the transcript assistant that answers questions 
about workshop content using retrieval augmented generation (RAG).
"""

import os
import re
import json
import time
import uuid
import sqlite3
import datetime
import tiktoken
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Change from relative to absolute import to support running as a script
try:
    # When running as a module (for Modal)
    from .utils import load_workshop_transcript
except ImportError:
    # When running as a script directly
    from utils import load_workshop_transcript

# === CONFIG SECTION ===
WORKSHOP_TRANSCRIPT_PATH = "/data/WS1-C2.vtt"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "workshop_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-3.5-turbo-16k"
TARGET_CHUNK_TOKEN_COUNT = 500
DEFAULT_MAX_CHUNKS = 5
DEFAULT_MAX_TOKENS = 12000
LOGS_DB_PATH = "logs.db"  # Path to SQLite database for logs

SYSTEM_PROMPT = """You are a helpful workshop assistant.
Answer questions based only on the workshop transcript sections provided.
If you don't know the answer or can't find it in the provided sections, say so."""

# === EMBEDDING FUNCTIONS ===

def get_openai_client():
    """Initialize and return an OpenAI client with API key from environment"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)

def generate_embedding(text: str) -> List[float]:
    """Generate an embedding vector for a text using OpenAI's API"""
    client = get_openai_client()
    
    # Request the embedding from OpenAI
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    
    # Extract the embedding vector from the response
    embedding = response.data[0].embedding
    
    return embedding

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(COMPLETION_MODEL)
        return len(encoding.encode(text))
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

# === CHUNKING FUNCTIONS ===

def extract_vtt_timestamps(transcript_path: str) -> List[Dict[str, Any]]:
    """Extract timestamps and speakers from VTT file"""
    segments = []
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Extract timestamps and text with regex
        # Format: 00:00:00.290 --> 00:00:01.350
        # hugo bowne-anderson: Everyone.
        pattern = r'(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+)\n(.*?)(?=\n\d+:\d+:\d+\.\d+|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for i, match in enumerate(matches):
            start_time, end_time, text = match
            
            # Extract speaker if available
            speaker = "Unknown"
            speaker_match = re.match(r'^([A-Za-z\s-]+):', text.strip())
            if speaker_match:
                speaker = speaker_match.group(1).strip()
            
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'timestamp': start_time,  # Use start time as the primary timestamp
                'text': text.strip(),
                'speaker': speaker
            })
        
        print(f"Extracted {len(segments)} segments from VTT file")
        if segments:
            print(f"Sample segment: {segments[0]}")
            
    except Exception as e:
        print(f"Error extracting timestamps: {str(e)}")
        
    return segments

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on multiple newlines or speaker changes"""
    # First, split on double newlines
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Further split paragraphs if they contain different speakers
    result = []
    for paragraph in paragraphs:
        # Check if paragraph has potential speaker markers (name followed by colon)
        speaker_changes = re.split(r'([A-Za-z\s-]+:)', paragraph)
        
        if len(speaker_changes) > 1:
            # Reassemble the speaker with their text
            current_text = ""
            speaker = ""
            
            for i, part in enumerate(speaker_changes):
                if i % 2 == 1:  # This is a speaker marker
                    if current_text:
                        result.append(current_text.strip())
                    speaker = part
                    current_text = speaker
                else:  # This is the text
                    current_text += part
            
            if current_text:
                result.append(current_text.strip())
        else:
            # No speaker changes, keep paragraph as is
            if paragraph.strip():
                result.append(paragraph.strip())
    
    return result

def create_chunks(text: str, target_token_count: int = TARGET_CHUNK_TOKEN_COUNT) -> List[Dict[str, Any]]:
    """Split text into chunks with metadata"""
    # Split text into paragraphs first
    paragraphs = split_into_paragraphs(text)
    
    chunks = []
    current_chunk_text = ""
    current_chunk_tokens = 0
    chunk_index = 0
    
    for paragraph in paragraphs:
        # Count tokens in this paragraph
        paragraph_tokens = count_tokens(paragraph)
        
        # If adding this paragraph would exceed our target, create a new chunk
        if current_chunk_tokens + paragraph_tokens > target_token_count and current_chunk_text:
            # Create a chunk with the text accumulated so far
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "text": current_chunk_text,
                "position": chunk_index,
                "token_count": current_chunk_tokens,
                "source": "workshop_transcript"
            }
            chunks.append(chunk)
            
            # Reset for new chunk
            current_chunk_text = paragraph
            current_chunk_tokens = paragraph_tokens
            chunk_index += 1
        else:
            # Add this paragraph to the current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n" + paragraph
            else:
                current_chunk_text = paragraph
            current_chunk_tokens += paragraph_tokens
    
    # Don't forget the last chunk if there's text left
    if current_chunk_text:
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "text": current_chunk_text,
            "position": chunk_index,
            "token_count": current_chunk_tokens,
            "source": "workshop_transcript"
        }
        chunks.append(chunk)
    
    return chunks

def chunk_workshop_transcript(transcript_path: str) -> List[Dict[str, Any]]:
    """Process a workshop transcript file and return chunks"""
    # Parse the VTT file to get accurate timestamps
    vtt_segments = extract_vtt_timestamps(transcript_path)
    
    # Load and clean the transcript for chunking
    transcript_text = load_workshop_transcript(transcript_path)
    
    # Create chunks from the transcript
    chunks = create_chunks(transcript_text)
    
    # Add timestamps to each chunk based on position
    for chunk in chunks:
        chunk_index = chunk['position']
        
        # Assign timestamps to chunks based on position
        if vtt_segments:
            # Distribute timestamps from segments across chunks
            segment_index = min(int(chunk_index * len(vtt_segments) / len(chunks)), len(vtt_segments) - 1)
            segment = vtt_segments[segment_index]
            
            # Add timestamp information
            timestamp = segment['timestamp']
            chunk['timestamp'] = timestamp
            
            # Include timestamp directly in the text
            # Format: [TIMESTAMP: 00:00:00]
            timestamp_marker = f"[TIMESTAMP: {timestamp}]\n"
            chunk['text'] = timestamp_marker + chunk['text']
            
            # Log timestamp assignment
            if chunk_index < 3:  # Just log a few for debugging
                print(f"Chunk {chunk_index} assigned timestamp: {timestamp}")
                
            # Extract speaker from segment
            if 'speaker' in segment:
                chunk['speaker'] = segment['speaker']
    
    return chunks

# === VECTOR STORAGE FUNCTIONS ===

def get_chroma_client():
    """Initialize and return a ChromaDB client with persistence"""
    # Create the directory if it doesn't exist
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    # Initialize ChromaDB with persistence
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def get_or_create_collection(client, collection_name=COLLECTION_NAME):
    """Get or create a collection in ChromaDB"""
    try:
        # First try to get the existing collection
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection '{collection_name}'")
    except:
        # If it doesn't exist, create a new one
        print(f"Creating new collection '{collection_name}'")
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Workshop transcript chunks"}
        )
    
    return collection

def add_chunks_to_collection(collection, chunks):
    """Add multiple chunks to the collection"""
    # Prepare data for storage
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for chunk in chunks:
        # Generate embedding for this chunk
        chunk['embedding'] = generate_embedding(chunk['text'])
        
        # Add to storage arrays
        ids.append(chunk['chunk_id'])
        documents.append(chunk['text'])
        embeddings.append(chunk['embedding'])
        
        # Create metadata
        metadata = {
            'position': chunk['position'],
            'token_count': chunk['token_count'],
            'source': chunk['source'],
            'timestamp': chunk.get('timestamp', 'Unknown'),
            'speaker': chunk.get('speaker', 'Unknown'),
        }
        metadatas.append(metadata)
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    return len(chunks)

def query_collection(collection, query_text, n_results=DEFAULT_MAX_CHUNKS):
    """Query the collection for relevant documents"""
    # Generate embedding for the query
    query_embedding = generate_embedding(query_text)
    
    # Search for similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

# === RETRIEVAL FUNCTIONS ===

def retrieve_relevant_chunks(question, collection_name=COLLECTION_NAME, n_results=DEFAULT_MAX_CHUNKS):
    """Retrieve chunks from vector database for a given question"""
    # Initialize client and get collection
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)
    
    # Query the collection
    results = query_collection(collection, question, n_results=n_results)
    
    # Process the results
    chunks = []
    if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
        # Create dummy distances (1.0) if not provided by the API
        distances = [1.0] * len(results['documents'][0])
        
        # Format chunks
        for i in range(len(results['documents'][0])):
            chunk = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'relevance': 1.0  # Default relevance since we don't have distances
            }
            chunks.append(chunk)
    
    return chunks

def combine_chunks(chunks, max_tokens=DEFAULT_MAX_TOKENS):
    """Combine multiple chunks into a single context, respecting token limit"""
    if not chunks:
        return ""
    
    # Sort chunks by position if available
    sorted_chunks = sorted(chunks, key=lambda x: int(x['metadata'].get('position', 0)))
    
    combined_text = ""
    total_tokens = 0
    
    for chunk in sorted_chunks:
        chunk_text = chunk['text']
        chunk_tokens = int(chunk['metadata'].get('token_count', 0))
        
        # If no token count in metadata, calculate it
        if chunk_tokens == 0:
            chunk_tokens = count_tokens(chunk_text)
        
        # Check if adding this chunk would exceed the token limit
        if total_tokens + chunk_tokens > max_tokens:
            break
        
        # Add separator between chunks if needed
        if combined_text:
            combined_text += "\n\n--- Next Section ---\n\n"
        
        # Add the chunk text
        combined_text += chunk_text
        total_tokens += chunk_tokens
    
    return combined_text

def format_sources(sources):
    """Format source information into a readable string"""
    if not sources:
        return "No source information available."
    
    formatted = "Sources (most similar first, lower distance = more similar):\n"
    for i, source in enumerate(sources):
        # Start with the source number and distance score
        source_header = f"{i+1}. "
        if source.get('relevance') is not None:
            source_header += f"[Distance: {source['relevance']:.4f}] "
        
        # Add position information
        position = source.get('position', 'Unknown')
        source_header += f"[Chunk {position}] "
        
        # Add speaker information if available
        speaker = source.get('speaker', 'Unknown')
        if speaker != 'Unknown':
            source_header += f"Speaker: {speaker}. "
        
        formatted += source_header + "\n"
        
        # Add the content with better formatting
        text = source.get('text', '')
        if text:
            # Format the text with proper indentation
            text_lines = text.split('\n')
            indented_text = '\n    '.join(text_lines)  # Indent continuation lines
            formatted += f"    {indented_text}\n\n"
    
    return formatted

def get_context_for_question(question, collection_name=COLLECTION_NAME, max_chunks=DEFAULT_MAX_CHUNKS):
    """Get relevant context from the vector database for a question"""
    # Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(question, collection_name, max_chunks)
    
    # Format source information
    sources = []
    for chunk in chunks:
        metadata = chunk['metadata']
        text = chunk['text']
        
        # Extract timestamp from text if it starts with [TIMESTAMP: ...]
        timestamp = metadata.get('timestamp', "Unknown")
        
        # Extract speaker
        speaker = metadata.get('speaker', "Unknown")
        
        source = {
            'position': metadata.get('position', 'Unknown'),
            'timestamp': timestamp,
            'speaker': speaker,
            'text': text,
            'relevance': chunk.get('relevance')
        }
        sources.append(source)
    
    # Combine chunks
    context = combine_chunks(chunks)
    
    return context, sources, chunks  # Return the raw chunks as well for logging

def process_workshop(transcript_path, collection_name=COLLECTION_NAME):
    """Process a workshop transcript and store it in the vector database"""
    # Create chunks from workshop transcript
    chunks = chunk_workshop_transcript(transcript_path)
    print(f"Created {len(chunks)} chunks from workshop transcript")
    
    # Initialize ChromaDB
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)
    
    # Check if collection already has documents
    try:
        count = collection.count()
        if count > 0:
            print(f"Collection '{collection_name}' already has {count} documents")
            return count
    except:
        # New collection, continue with adding chunks
        pass
    
    # Add all chunks to the collection
    num_added = add_chunks_to_collection(collection, chunks)
    print(f"Added {num_added} chunks to collection '{collection_name}'")
    
    return len(chunks)

# === LOGGING FUNCTIONS ===

def init_logs_db():
    """Initialize the SQLite database for logging with a single table"""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(LOGS_DB_PATH) if os.path.dirname(LOGS_DB_PATH) else '.', exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(LOGS_DB_PATH)
    cursor = conn.cursor()
    
    # Create a single logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        question TEXT,
        response TEXT,
        num_chunks INTEGER,
        context_tokens INTEGER,
        completion_tokens INTEGER,
        embedding_tokens INTEGER,
        total_tokens INTEGER,
        latency REAL,
        model TEXT,
        sources TEXT,
        success INTEGER,
        feedback_rating TEXT,
        feedback_reason TEXT,
        feedback_notes TEXT,
        feedback_user TEXT
    )
    ''')
    
    # Attempt to add new columns if they don't exist (for backward compatibility)
    feedback_columns = {
        'feedback_rating': 'TEXT',
        'feedback_reason': 'TEXT',
        'feedback_notes': 'TEXT',
        'feedback_user': 'TEXT'
    }
    
    for column, col_type in feedback_columns.items():
        try:
            cursor.execute(f"ALTER TABLE logs ADD COLUMN {column} {col_type}")
            print(f"Added column '{column}' to logs table.")
        except sqlite3.OperationalError as e:
            # Ignore error if column already exists
            if not 'duplicate column name' in str(e):
                print(f"Warning: Could not add column '{column}': {e}")

    conn.commit()
    conn.close()
    
    print(f"Initialized logs database at {LOGS_DB_PATH}")
    return True

def log_interaction(question, response, context_info, model, start_time, end_time, success=True):
    """Log an interaction to the single table database"""
    log_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    # Calculate latency
    latency = end_time - start_time # Use end_time for accurate latency
    
    # Extract token information
    context_tokens = context_info.get("context_tokens", 0)
    completion_tokens = context_info.get("completion_tokens", 0)
    embedding_tokens = context_info.get("embedding_tokens", 0)
    total_tokens = context_tokens + completion_tokens + embedding_tokens
    
    # Serialize source information to JSON
    sources_json = json.dumps([{
        'id': chunk.get('id', ''),
        'position': chunk.get('metadata', {}).get('position', 0),
        'relevance': chunk.get('relevance', 0.0),
        'tokens': chunk.get('metadata', {}).get('token_count', 0),
        'timestamp': chunk.get('metadata', {}).get('timestamp', ''),
        'speaker': chunk.get('metadata', {}).get('speaker', ''),
        'text': chunk.get('text', '')  # Include the full text of the source
    } for chunk in context_info.get("chunks", [])])
    
    # Connect to database
    conn = sqlite3.connect(LOGS_DB_PATH)
    cursor = conn.cursor()
    
    # Insert log record
    cursor.execute(
        '''
        INSERT INTO logs (
            id, timestamp, question, response, num_chunks, context_tokens, 
            completion_tokens, embedding_tokens, total_tokens, latency, model,
            sources, success
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            log_id, timestamp, question, response, context_info.get("num_chunks", 0),
            context_tokens, completion_tokens, embedding_tokens, total_tokens,
            latency, model, sources_json, 1 if success else 0
        )
    )
    
    conn.commit()
    conn.close()
    
    return log_id

def log_feedback(log_id: str, rating: str, reason: str, notes: str, user: str):
    """Update an existing log entry with feedback details."""
    if not log_id:
        print("Error: No log_id provided for feedback.")
        return False
        
    if not reason: # Ensure reason is provided
        print("Error: Feedback reason cannot be empty.")
        return False
        
    try:
        conn = sqlite3.connect(LOGS_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            UPDATE logs 
            SET feedback_rating = ?, 
                feedback_reason = ?, 
                feedback_notes = ?, 
                feedback_user = ?
            WHERE id = ?
            ''',
            (rating, reason, notes, user, log_id)
        )
        
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        if rows_affected == 0:
            print(f"Warning: Feedback submitted but no log found with ID: {log_id}")
            return False
        else:
            print(f"Feedback successfully logged for ID: {log_id}")
            return True
            
    except Exception as e:
        print(f"Error logging feedback for ID {log_id}: {e}")
        # Optionally re-raise or handle differently
        if conn:
            conn.close()
        return False

def get_recent_logs(limit=5):
    """Get the most recent logs"""
    conn = sqlite3.connect(LOGS_DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute(
        '''
        SELECT id, timestamp, question, num_chunks, context_tokens, 
               total_tokens, latency
        FROM logs
        ORDER BY timestamp DESC
        LIMIT ?
        ''',
        (limit,)
    )
    
    # Convert to list of dictionaries
    logs = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return logs

# === MAIN FUNCTIONS ===

def answer_question(question):
    """
    Answers a question based on the workshop transcript
    Returns both the answer and context information
    """
    print(f"Retrieved existing collection '{COLLECTION_NAME}'")
    
    # Get relevant context from the vector database
    context, sources, chunks = get_context_for_question(
        question=question,
        collection_name=COLLECTION_NAME,
        max_chunks=DEFAULT_MAX_CHUNKS
    )
    
    # Log information about the context
    context_tokens = count_tokens(context)
    num_chunks = len(sources)
    print(f"Retrieved {context_tokens} tokens of relevant context")
    print(f"Retrieved {num_chunks} source chunks")
    
    start_time = time.time()  # Start timing from before API call
    
    # Generate a response from the LLM using the context
    client = get_openai_client()
    try:
        # Make the API call
        response = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Workshop Transcript Sections:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        # Get token usage
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else context_tokens
    
        # Format source information
        source_text = format_sources(sources)
        
        # Record metrics
        context_info = {
            "num_chunks": num_chunks,
            "context_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "embedding_tokens": num_chunks * 1536,  # Estimate based on embedding dimensions
            "chunks": chunks
        }
        
        return answer, source_text, context_info
    
    except Exception as e:
        error_message = f"Sorry, an error occurred: {str(e)}"
        
        # Still log the error
        context_info = {
            "num_chunks": num_chunks,
            "context_tokens": context_tokens,
            "completion_tokens": 0,
            "embedding_tokens": num_chunks * 1536,
            "chunks": chunks
        }
        
        return error_message, context_info

def main():
    """Main entry point for the simple RAG system"""
    import sys
    
    print("Simple RAG System for Workshop Q&A")
    print("----------------------------------")
    print(f"Workshop transcript: {WORKSHOP_TRANSCRIPT_PATH}")
    
    # Initialize the database if needed
    init_logs_db()
    
    # Set up the workshop content
    print("Ensuring workshop content is processed and stored...")
    num_chunks = process_workshop(WORKSHOP_TRANSCRIPT_PATH, COLLECTION_NAME)
    print(f"Workshop has been processed into {num_chunks} searchable chunks")
    
    if len(sys.argv) > 1:
        # Use the command line argument as the question
        custom_question = " ".join(sys.argv[1:])
        print(f"\nQuestion: {custom_question}")
        
        start_time = time.time()
        response, _, context_info = answer_question(custom_question)
        
        print(f"\nAnswer: {response}")
        
        # Log the interaction
        log_interaction(custom_question, response, context_info, COMPLETION_MODEL, start_time, time.time())
        
    else:
        # Run the predefined test questions
        test_questions = [
            "What is the background of the presenters?",
            "What is this workshop about?",
            "What are the main topics covered in this workshop?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n[TEST {i}] Question: {question}")
            
            start_time = time.time()
            response, _, context_info = answer_question(question)
            
            print(f"\n[TEST {i}] Answer: {response}")
            
            # Log the interaction
            log_interaction(question, response, context_info, COMPLETION_MODEL, start_time, time.time())
        
        # Print recent logs summary
        print("\nMost recent query logs:")
        logs = get_recent_logs(3)
        for log in logs:
            log_time = log.get("timestamp", "")
            question = log.get("question", "")[:30] + "..." if len(log.get("question", "")) > 30 else log.get("question", "")
            tokens = log.get("total_tokens", 0)
            latency = log.get("latency", 0)
            print(f"Time: {log_time}, Question: {question}, Tokens: {tokens}, Latency: {latency:.2f}s")
    
    print("\nTest questions complete. Script execution finished.")

if __name__ == "__main__":
    main() 