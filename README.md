# Standalone RAG + Feedback App (Modal Deployable)

This repository contains a simplified implementation of a Retrieval Augmented Generation (RAG) system with user feedback capabilities, deployable using Modal. **It serves as the starting point for exploring Evaluation-Driven Development (EDD) in the context of building LLM applications.**

## Features

- Core RAG functionality:
  - VTT transcript parsing (default: `data/WS1-C2.vtt`)
  - Text chunking
  - Vector storage with ChromaDB
  - Semantic search
  - Response generation
  - Source citation
- SQLite logging system (`logs.db`) capturing interactions and feedback.
- **Persistent Logging:** Logs are stored persistently across deployments using a `modal.Volume` named `rag-app-logs`.
- **User Feedback:** Gradio interface allows users to rate responses (PASS/FAIL) and provide reasons/notes.
- Modal deployment configuration (`src/app-modal.py`) including:
  - Web interface using Gradio named `query-workshop`
  - API endpoints with FastAPI (including `/logs`)
  - Persistent volume setup
  - Cloud deployment readiness

## Learning Objectives & Project Goal

This application serves as a practical exercise for the course, focusing on **Evaluation-Driven Development (EDD)** when building LLM applications.

We're starting with a basic Retrieval Augmented Generation (RAG) system that allows you to query a workshop transcript. The initial goal is hands-on evaluation:

1.  **Interact:** Ask questions about the workshop content.
2.  **Evaluate:** Use the Gradio interface to provide feedback (Pass/Fail, reason, notes) based on your initial assessment of the response quality. Think of this as a "vibe check" – does the answer seem helpful and relevant?
3.  **Log:** Your feedback is logged, forming the basis for more rigorous evaluation later.

This first version is intentionally simple. As we progress through the course, we will build upon this foundation:

*   Develop a more formal **evaluation harness** based on the collected feedback and defined metrics (like faithfulness, relevance, etc.).
*   Explore adding **more sophisticated functionality** to the RAG pipeline.
*   Consider advanced patterns, such as creating **agents** that could potentially search external resources or use tools if the answer isn't found directly in the transcript.

The aim is to iteratively improve the application, guided by continuous evaluation – the core principle of EDD.

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
4. **(Optional) Prepare Custom Data:** The current default transcript is `data/WS1-C2.vtt`. To use a different transcript (e.g., `data/WS1.vtt` or your own), place the file in the `data/` directory and update the `WORKSHOP_TRANSCRIPT_PATH` variable within `src/simple_rag.py` to point to the correct file path (e.g., `/data/your_file.vtt`).

## Usage

### Deploy with Modal

1. **Install Modal CLI:**
   ```bash
   pip install modal
   ```
2. **Set up Modal account and API secret:**
   ```bash
   modal setup
   modal secret create openai-secret OPENAI_API_KEY=<your-api-key-from-.env>
   ```
3. **Serve locally for development:**
   ```bash
   modal serve -m src.app-modal
   ```
   *(Access the Gradio UI at the provided localhost URL)*
4. **Deploy to Modal cloud:**
   ```bash
   modal deploy -m src.app-modal
   ```
   *(Access the Gradio UI and `/logs` endpoint at the provided `query-workshop...modal.run` URL)*

## Troubleshooting

- **API Key Issues**: Ensure `.env` exists and contains a valid key. Make sure the `openai-secret` is created correctly in Modal.
- **Module Not Found**: Ensure dependencies are installed using `pip install -r requirements.txt`. Ensure commands use the `-m src.app-modal` flag correctly if running Modal commands.
- **Modal Issues**: Refer to the [Modal documentation](https://modal.com/docs). 