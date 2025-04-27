# Transcript Agent MVP

This directory contains the code for the transcript processing agent.

## Purpose

This agent is designed to process transcripts, potentially interact with an LLM (like GPT-4o mini via OpenAI), and use tools (like a database query tool) to answer questions or perform tasks based on the transcript content. The primary script is `agent.py`.

## How to Run

Currently, the execution details need refinement. The agent logic is within `agent.py`, but the previous command-line interface (`run_agent_cli.py`) has been removed. Execution might involve directly running or importing functions from `agent.py`.

*(Placeholder: Update with specific command or instructions once finalized)*

## Logging

Agent execution details, including LLM interactions (inputs, outputs, tool calls) and errors, are logged sequentially in JSON format to `agent_run_log.json` within this directory. This log is currently checked into Git for educational/demonstration purposes.

## Evaluation and Annotation Retrieval (Current Challenge)

A key goal is to evaluate the agent's performance using manual annotations (e.g., 'Tool Usage Correctness', 'SQL Correctness', 'Final Answer Quality') applied to spans via the Arize Phoenix UI.

**Challenge:** We are currently facing difficulty retrieving these specific manual annotations programmatically using the Phoenix Python SDK (`arize-phoenix` library).

**What We've Tried:**
1.  `client.get_spans_dataframe(project_name='transcript-agent-mvp')`: This method successfully retrieves span data but does not include the custom UI annotations in the resulting DataFrame columns.
2.  `client.get_trace_dataset(project_name='transcript-agent-mvp')`: This method returns a `TraceDataset` object.
    *   The `trace_dataset.dataframe` attribute contains the same span data as above (no UI annotations).
    *   The `trace_dataset.evaluations` attribute, which seemed relevant, returns an empty list (`[]`).

**Current Status:**
*   We have confirmed the annotations *are* visible and stored correctly within the Phoenix UI.
*   We have communicated these findings to the Arize Phoenix team via Slack and are awaiting further guidance on the correct SDK method or procedure to access these UI annotations, acknowledging they might be treated as separate entities from spans in the backend data model.
*   The Phoenix team mentioned an upcoming overhaul related to annotations (Ref: GitHub Issue #5917).

*(This README reflects the status as of [Insert Date/Time]).* 