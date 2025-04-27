import sqlite3
import re
import os
from pathlib import Path

# --- Constants ---
# Assuming the script is run from the root of the project directory
VTT_FILE_PATH = Path("data/WS1-C2.vtt")
DB_FILE_PATH = Path("data/workshop1_transcript.db")
TABLE_NAME = "transcript_segments"
SESSION_NAME = "WS1-C2"

# --- Helper Functions ---

def parse_timestamp(timestamp_str: str) -> float:
    """Converts HH:MM:SS.ms timestamp string to total seconds."""
    try:
        h, m, s_ms = timestamp_str.split(':')
        s, ms = s_ms.split('.')
        total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
        return total_seconds
    except ValueError:
        print(f"Warning: Could not parse timestamp: {timestamp_str}")
        return 0.0 # Return 0 or handle error as appropriate

def extract_speaker_and_text(text_line: str) -> (str | None, str):
    """Extracts speaker name (if present) and the text content."""
    match = re.match(r"([^:]+):\s*(.*)", text_line)
    if match:
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        # Handle potential abbreviations like 'gp:'
        if len(speaker) <= 3 and not speaker.isnumeric(): # Simple heuristic
             # Could add more sophisticated speaker normalization here if needed
             pass
        return speaker, text
    else:
        # No speaker found, assume it's continuation or speakerless
        return None, text_line.strip()

def calculate_word_count(text: str) -> int:
    """Calculates the number of words in a text string."""
    return len(text.split())

# --- Main Database Creation Function ---

def create_database():
    """Parses the VTT file and creates/populates the SQLite database."""
    
    print(f"Checking for existing database at {DB_FILE_PATH}...")
    if DB_FILE_PATH.exists():
        print("Database found. Deleting existing database.")
        DB_FILE_PATH.unlink() # Remove the existing file

    print(f"Creating new database at {DB_FILE_PATH}...")
    # Ensure the data directory exists
    DB_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = None # Initialize connection variable
    try:
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()

        print(f"Creating table '{TABLE_NAME}'...")
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            segment_id INTEGER PRIMARY KEY,
            session_name TEXT NOT NULL,
            start_time_seconds REAL NOT NULL,
            end_time_seconds REAL NOT NULL,
            speaker TEXT, 
            text TEXT NOT NULL,
            word_count INTEGER NOT NULL
        )
        """)
        conn.commit()
        print("Table created successfully.")

        print(f"Reading VTT file from {VTT_FILE_PATH}...")
        if not VTT_FILE_PATH.exists():
             print(f"Error: VTT file not found at {VTT_FILE_PATH}")
             return

        with open(VTT_FILE_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print("Parsing VTT file and inserting data...")
        current_segment_id = None
        current_start_time = None
        current_end_time = None
        current_text_lines = []
        segments_processed = 0

        # Regex patterns
        segment_id_pattern = re.compile(r"^(\d+)$")
        timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})")
        
        # Iterate through lines, processing segments
        for line in lines:
            line = line.strip()

            # Skip header and empty lines
            if not line or line == 'WEBVTT':
                continue

            # Check for segment ID
            id_match = segment_id_pattern.match(line)
            if id_match:
                 # If we have pending text from previous segment, process it
                 if current_segment_id is not None and current_text_lines:
                     speaker, text = extract_speaker_and_text(" ".join(current_text_lines))
                     word_count = calculate_word_count(text)
                     cursor.execute(f"""
                     INSERT INTO {TABLE_NAME} (segment_id, session_name, start_time_seconds, end_time_seconds, speaker, text, word_count)
                     VALUES (?, ?, ?, ?, ?, ?, ?)
                     """, (current_segment_id, SESSION_NAME, current_start_time, current_end_time, speaker, text, word_count))
                     segments_processed += 1
                     if segments_processed % 100 == 0:
                         print(f"Processed {segments_processed} segments...")
                 
                 # Start new segment
                 current_segment_id = int(id_match.group(1))
                 current_text_lines = []
                 current_start_time = None
                 current_end_time = None
                 continue # Move to the next line

            # Check for timestamp line (only if segment ID is known)
            if current_segment_id is not None and current_start_time is None:
                 ts_match = timestamp_pattern.match(line)
                 if ts_match:
                     current_start_time = parse_timestamp(ts_match.group(1))
                     current_end_time = parse_timestamp(ts_match.group(2))
                     continue # Move to the next line

            # Assume it's a text line (only if timestamp is known)
            if current_start_time is not None:
                 current_text_lines.append(line)

        # Process the very last segment after the loop ends
        if current_segment_id is not None and current_text_lines:
            speaker, text = extract_speaker_and_text(" ".join(current_text_lines))
            word_count = calculate_word_count(text)
            cursor.execute(f"""
            INSERT INTO {TABLE_NAME} (segment_id, session_name, start_time_seconds, end_time_seconds, speaker, text, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (current_segment_id, SESSION_NAME, current_start_time, current_end_time, speaker, text, word_count))
            segments_processed += 1
            print(f"Processed {segments_processed} segments...")

        conn.commit()
        print(f"\nDatabase '{DB_FILE_PATH}' created and populated successfully with {segments_processed} segments.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except FileNotFoundError:
        print(f"Error: Input VTT file not found at {VTT_FILE_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

# --- Execution Block ---

if __name__ == "__main__":
    create_database() 