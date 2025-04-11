import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Simple function to estimate token count
def estimate_tokens(text):
    """Rough estimate of token count - 1 token is roughly 4 chars in English."""
    return len(text) // 4

def init_openai_client():
    """Initialize and return the OpenAI client."""
    # Load environment variables
    load_dotenv()
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")
    
    # Initialize the OpenAI client
    return OpenAI(api_key=api_key)

def clean_vtt_transcript(content):
    """Clean VTT format to extract just the spoken text."""
    lines = content.strip().split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        # Skip the WEBVTT header
        if i == 0 and lines[i].strip() == "WEBVTT":
            i += 1
            continue
        
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
        
        # Skip timestamp lines (they typically contain "-->")
        if "-->" in lines[i]:
            i += 1
            continue
        
        # Skip index numbers that appear alone on lines
        if re.match(r'^\d+$', lines[i].strip()):
            i += 1
            continue
        
        # This should be a line of actual transcript
        cleaned_lines.append(lines[i])
        i += 1
    
    return "\n".join(cleaned_lines)

def load_workshop_transcript(file_path):
    """Load and clean the workshop transcript from a VTT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return clean_vtt_transcript(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading transcript: {str(e)}") 