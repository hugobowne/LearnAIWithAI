# Standalone RAG + Feedback App (Modal Deployable)

This repository contains a simplified implementation of a Retrieval Augmented Generation (RAG) system with user feedback capabilities, deployable using Modal.

## Features

- Core RAG functionality:
  - VTT transcript parsing (`data/WS1.vtt`)
  - Text chunking
  - Vector storage with ChromaDB
  - Semantic search
  - Response generation
  - Source citation
- SQLite logging system (`logs.db`) capturing interactions and feedback.
- **Persistent Logging:** Logs are stored persistently across deployments using a `modal.Volume` named `rag-app-logs`.
- **User Feedback:** Gradio interface allows users to rate responses (PASS/FAIL) and provide reasons/notes.
- Modal deployment configuration (`app-modal.py`) including:
  - Web interface using Gradio
  - API endpoints with FastAPI (including `/logs`)
  - Persistent volume setup
  - Cloud deployment readiness

## Setup

1. **Clone the repository.**
2. **Create a `.env` file** in the repository root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have Python 3.9+ installed)*
4. **(Optional) Prepare Custom Data:** Replace `data/WS1.vtt` with your own VTT transcript if desired. The code expects the file at this location.

## Usage

### Run Locally (Limited Functionality)

Running `python simple_rag.py` locally will test the core RAG pipeline and logging (creating a local `logs.db`), but it won't use the Gradio UI or persistent Modal features.

### Deploy with Modal

1. **Install Modal CLI:**
   ```bash
   pip install modal-client
   ```
2. **Set up Modal account and API secret:**
   ```bash
   modal setup
   modal secret create openai-secret OPENAI_API_KEY=<your-api-key-from-.env>
   ```
3. **Serve locally for development:**
   ```bash
   modal serve app-modal.py
   ```
   *(Access the Gradio UI at the provided localhost URL)*
4. **Deploy to Modal cloud:**
   ```bash
   modal deploy app-modal.py
   ```
   *(Access the Gradio UI and `/logs` endpoint at the provided `.modal.run` URL)*

## Troubleshooting

- **API Key Issues**: Ensure `.env` exists and contains a valid key. Make sure the `openai-secret` is created correctly in Modal.
- **Module Not Found**: Ensure dependencies are installed using `pip install -r requirements.txt`.
- **Modal Issues**: Refer to the [Modal documentation](https://modal.com/docs). 