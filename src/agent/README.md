# Evaluating a Transcript Query Agent 🧪

This directory provides the code and resources for a workshop module focused on **evaluating LLM-based agents**. We use a simple agent designed to query lecture transcripts as a practical case study.

## Workshop Goal: Agent Evaluation 🎯

The primary goal of this module is to understand and implement techniques for evaluating the performance of AI agents, particularly those that use tools (like database queries).

Key learning objectives include:
*   Defining relevant evaluation metrics for agent behavior (e.g., tool usage, intermediate step correctness, final output quality).
*   Establishing a test set with ground truth data.
*   Building an evaluation harness to automate performance assessment.
*   Experimenting with agent configurations (e.g., different LLMs) and measuring the impact on evaluation results.

## Case Study: Transcript Agent MVP 🤖

To explore these evaluation concepts, we use a Minimum Viable Product (MVP) agent. This agent serves as the first step towards a broader vision of a data science agent capable of using multiple tools (SQL querying, data analysis, data visualization). For this initial stage and evaluation focus, the MVP is:
*   An LLM (like GPT-4o mini) given a task: answer questions about a workshop transcript.
*   Equipped with a single tool: a SQL query tool (`tools.py`) to retrieve information from a transcript database (`data/workshop1_transcript.db`, created by `create_transcript_db.py`).
*   Implemented primarily in `agent.py`.
*   Instrumented using Arize Phoenix (`arize-phoenix`) for observability, allowing us to capture traces of its execution (LLM calls, tool usage, etc.).

This simple agent provides a concrete target for developing and applying our evaluation strategies.

## Project Goal

The broader vision is to build a data science agent capable of:
1.  Executing SQL queries against a database.
2.  Performing data analysis.
3.  Generating data visualizations.

This MVP focuses on the first capability (SQL queries) using lecture transcript data as a relevant domain for evaluation within an educational context.

## Purpose (MVP)

This agent represents a basic implementation: an LLM (like GPT-4o mini via OpenAI) equipped with a single tool – a SQL query tool defined in `tools.py`. The agent's goal is to answer questions about workshop transcript data by querying a database (created using `create_transcript_db.py`).

The primary agent logic resides in `agent.py`. Importantly, the agent execution is instrumented using the Arize Phoenix library (`arize-phoenix`) to send trace data (LLM inputs/outputs, tool calls/results, etc.) to the Arize platform. This allows for observability, debugging, and evaluation of the agent's performance.

## Evaluation Strategy 📊

A key goal of this MVP is to establish a robust evaluation framework for agent performance. We plan to evaluate based on:
1.  **Tool Usage Correctness:** Did the agent choose the correct tool (i.e., the SQL query tool)?
2.  **SQL Correctness:** Was the generated SQL query syntactically and semantically correct for the question?
3.  **Final Answer Quality:** Was the final answer derived from the SQL result accurate and relevant?

To bootstrap this process before launch, we are using synthetic data generation. Test cases, including personas, scenarios, and specific questions (informed by `docs/AGENT_SPEC.md` and exemplified in `docs/test_queries.json`), are being generated synthetically and then hand-labeled. This approach helps us develop a Minimum Viable Evaluation (MVE) harness and kickstart the data flywheel.

## Transcript Database 💾

To support the SQL query functionality, a simple SQLite database (`data/workshop1_transcript.db`) is created using the `src/agent/create_transcript_db.py` script. This script parses a VTT transcript file (`data/WS1-C2.vtt`) and populates a `transcript_segments` table.

The goal was to create a minimal database structure sufficient for querying basic information about the workshop transcript, such as identifying segments by speaker, time, or content. The table includes columns like `segment_id`, `session_name`, `start_time_seconds`, `end_time_seconds`, `speaker`, `text`, and `word_count`.

## How to Run ▶️

The agent logic is within `agent.py`.

## Logging 📝

Agent execution details, including LLM interactions (inputs, outputs, tool calls) and errors, are logged sequentially in JSON format to `agent_run_log.json` within this directory. 

## Accessing Manual Annotations for Evaluation 🏷️

For this workshop, we will evaluate the agent's performance using manual annotations (e.g., 'Tool Usage Correctness', 'SQL Correctness', 'Final Answer Quality') that were previously added to agent traces via the Arize Phoenix UI.

To access this data for our evaluation harness, we use a dataset that was manually exported from the Phoenix UI as a CSV file. The `src/agent/eval/parse_spans.ipynb` notebook is responsible for loading and parsing this CSV, making both the agent's execution data and the corresponding human annotations available for analysis.

*(Note: While direct programmatic access via SDK is ideal for production, using the CSV export provides a stable dataset for this educational module focused on evaluation techniques.)*

## Evaluation Harness Development 🛠️

With the annotated data extracted (via the CSV workaround and `parse_spans.ipynb`), the next step is to build an evaluation harness using the `src/agent/eval/run_evaluation.ipynb` notebook.

The goals for this harness are:
1.  **Define Metrics:** Formalize the evaluation criteria based on the manual annotations ('Tool Usage Correctness', 'SQL Correctness', 'Final Answer Quality').
2.  **Implement Evaluators:** Create functions or methods to automatically assess agent performance against these metrics. This may involve techniques like:
    *   Exact matching or rule-based checks for tool usage and SQL syntax.
    *   Using LLMs-as-judges for more nuanced evaluations (e.g., assessing the quality and correctness of the final generated answer based on the retrieved database context).
3.  **Run Experiments:** Structure the harness to easily run the agent's test queries through it and calculate aggregate performance scores.
4.  **Compare Configurations:** Enable experimentation by making it easy to swap components (e.g., different LLM models used by the agent) and compare their performance using the established evaluation metrics.

This harness will provide a quantitative way to measure agent performance and guide further development. 