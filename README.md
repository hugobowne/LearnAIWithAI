# Build AI to Learn AI: Workshop Q&A Examples

Welcome! 👋 Get ready for an exciting part of the **Building LLM Applications for Data Scientists and Software Engineers** course! This repository provides starting code for a unique learning experience: **you'll build your own AI application using transcripts from our AI workshops.** The goal is for you to actively build upon, modify, and experiment with this code using the concepts you learn throughout the course.

Think of it like this: you're building AI to learn *about* AI! You can even query the Q&A system you build about the workshop content to deepen your understanding.

The repository contains two initial examples to get you started:
*   A basic command-line Q&A script (`src/basic_qa.py`) - **This is your recommended starting point!**
*   A more advanced Retrieval Augmented Generation (RAG) application (`src/simple_rag.py` & `src/app-modal.py`).

## Learning Objectives & Project Goal

The primary goal is to provide practical exercises focusing on **Evaluation-Driven Development (EDD)** and iterative improvement when building LLM applications.

**We recommend starting with the `src/basic_qa.py` script.** This simple example demonstrates the core interaction with an LLM for Q&A based on a provided text document. It serves as a foundation for applying concepts learned early in the course.

**The more advanced RAG application (`src/simple_rag.py`, `src/app-modal.py`)** showcases techniques like document chunking, retrieval, and feedback logging, which address the limitations of the basic approach, especially for longer documents. You can explore this application as we cover RAG techniques.

The aim is to start simply, iterate, and build understanding, guided by continuous evaluation – the core principle of EDD.

## Features

### 1. Basic Q&A Script (`src/basic_qa.py`)
- Command-line interface (takes question as argument).
- Loads the full workshop transcript (`data/WS1-C2.vtt`).
- **Limitation:** Truncates the transcript if it exceeds the LLM's context window (approx. 60k characters), printing a warning. Answers are based only on the initial part of the text in such cases.
- Uses OpenAI API (`gpt-3.5-turbo-16k`) to answer based on the (potentially truncated) text.
- Requires a `.env` file for the OpenAI API key.

### 2. RAG + Feedback App (`src/simple_rag.py`, `src/app-modal.py`)
- Core RAG functionality overcoming basic script limitations:
  - VTT transcript parsing (`data/WS1-C2.vtt` by default).
  - Text chunking (to fit within context limits).
  - Vector storage with ChromaDB & semantic search (retrieval).
  - Response generation based on relevant chunks.
  - Source citation.
- SQLite logging system (`logs.db`) capturing interactions and user feedback.
- **Persistent Logging:** Uses `modal.Volume` for persistent logs across deployments.
- **User Feedback:** Gradio interface allows users to rate responses (PASS/FAIL) and provide reasons/notes for evaluation.
- Modal deployment configuration (`src/app-modal.py`) including:
  - Web interface (`query-workshop`).
  - API endpoints (`/logs`).
  - Persistent volume setup.

## Setup

1.  **Clone the repository.**
2.  **Create a `.env` file** in the repository root and add your OpenAI API key:
    ```
    OPENAI_API_KEY=your-key-here
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have Python 3.9+ installed)*
4.  **Data:** The `data/` directory contains the workshop transcript `WS1-C2.vtt`.
    *   The `basic_qa.py` script reads this file directly.
    *   The RAG app (`simple_rag.py`) uses this file by default for chunking and indexing. You can (optionally) modify the `WORKSHOP_TRANSCRIPT_PATH` variable within `src/simple_rag.py` if you wish to process a different custom transcript file you place in `data/`.

## Usage

### 1. Getting Started: Basic Q&A Script (`src/basic_qa.py`)

This is the recommended starting point for course exercises.

-   **Run from the repository root:**
    ```bash
    python src/basic_qa.py "Your question about the workshop?"
    ```
-   **How it works:** It loads the transcript, checks if it's too long for the LLM, truncates it with a warning if necessary, and then asks the LLM to answer your question based on the available text.
-   **Limitations:** Because it loads the entire text (or the first ~60k characters), it cannot effectively use very long documents and lacks the precision of RAG. This limitation motivates the need for more advanced techniques.
-   **Your Turn:** Use this script to experiment! Try different questions, modify the system prompt, explore different models, or even try implementing basic error handling or simple chunking strategies based on what you learn in the course.

### 2. Exploring the RAG Application (`src/simple_rag.py` & `src/app-modal.py`)

This application demonstrates a more robust RAG approach suitable for longer documents and includes a web UI for interaction and feedback.

-   **Install Modal CLI:**
    ```bash
    pip install modal
    ```
-   **Set up Modal account and API secret:**
    ```bash
    modal setup
    modal secret create openai-secret OPENAI_API_KEY=<your-key-from-.env>
    ```
-   **Serve locally for development:**
    ```bash
    modal serve -m src.app-modal
    ```
    *(Access the Gradio UI at the provided localhost URL)*
-   **Deploy to Modal cloud:**
    ```bash
    modal deploy -m src.app-modal
    ```
    *(Access the Gradio UI and `/logs` endpoint at the provided `query-workshop...modal.run` URL)*

## Troubleshooting

-   **API Key Issues**: Ensure `.env` exists and contains a valid key (`OPENAI_API_KEY=...`). For the RAG app, ensure the `openai-secret` is created correctly in Modal.
-   **Module Not Found**: Ensure dependencies are installed (`pip install -r requirements.txt`). Ensure Modal commands use the `-m src.app-modal` flag correctly.
-   **File Not Found (basic_qa.py)**: Ensure you are running the `python src/basic_qa.py ...` command from the repository root directory (`LearnAIWithAI/`).
-   **Modal Issues**: Refer to the [Modal documentation](https://modal.com/docs). 