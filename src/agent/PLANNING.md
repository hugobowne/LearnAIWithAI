# Agent Observability and Evaluation Plan

This document outlines the plan for adding observability and building an evaluation harness for the transcript agent.

**Goal:** Implement a system to monitor the agent's performance, create a ground-truth dataset, build an evaluation harness, and use it to compare different agent configurations (e.g., different LLMs).

**Steps:**

1.  **Integrate Observability:**
    *   Modify `agent.py` (and potentially `tools.py`) to log key interaction data (user query, system prompt, messages, tool calls/responses, final answer, errors) to an observability platform.
    *   *Note:* The target platform is BrainTrust.

2.  **Create Labeled Test Set:**
    *   Use the initial test queries (`docs/test_queries.json`) and their logged results in the observability platform.
    *   Hand-label these results within the platform to create a golden dataset (ground truth) for evaluation.

3.  **Develop Basic Evaluation Harness:**
    *   Create a script (`src/agent/braintrust/evaluate.py`) that uses the BrainTrust Dataset created from the labeled logs (`SQL-agent-annotated`) as input.
    *   Implement a basic evaluation method (e.g., LLM-as-judge, potentially comparing agent outputs against the labeled answers).
    *   The script should run the agent over the test set and calculate relevant metrics.

4.  **Compare Models/Configurations:**
    *   Use the evaluation harness to compare the performance of the current model (`gpt-4o-mini`) against another model (e.g., `gpt-4o`).
    *   This involves running the harness with each model configuration and comparing the resulting evaluation metrics. 

## Minimum Viable Instrumentation (MVI) Plan (for Step 1)

Goal: Capture the minimum data needed in BrainTrust to allow feedback on: 
1. Tool Choice (Was `query_database` called?)
2. Tool Input (Was the generated SQL appropriate?)
3. Final Generation (Was the final answer good?)

Approach:

1.  **Initialize Logger:** Add `braintrust.init_logger(...)` at the beginning of `agent.py`.
2.  **Wrap OpenAI Client:** Modify the OpenAI client initialization to use `braintrust.wrap_openai(...)`.
3.  **Trace Agent Function:** Add the `@braintrust.traced` decorator to the `run_agent_conversation` function definition.

This approach relies on the automatic logging provided by `wrap_openai` (for LLM call details including tool calls/SQL) and `@traced` (for overall function inputs/outputs) to capture the necessary data points without manual span creation initially. 

## Initial Test Run (15 Queries) - Observations

*   Successfully ran `agent.py` after implementing the MVI and fixing the database path issue in `tools.py`.
*   15 traces corresponding to the expanded test queries (`docs/test_queries.json`) were successfully logged to the BrainTrust "Transcript Agent" project.
*   Detailed traces, including inputs, outputs, LLM calls (with tool calls/SQL), and final answers, are **viewable in the BrainTrust UI under the "Logs" tab**.
*   **Initial Agent Behavior Notes:**
    *   Handled basic queries, off-topic refusals, prompt injection attempts, and complex filtering reasonably well.
    *   Successfully ignored SQL injection characters and executed only the valid part of the query.
    *   Attempted to handle complex/conflicting instructions in the "wild" query by decomposing it into multiple tool calls.
    *   Failed on the `SUM()` aggregation query (`total_time` returned `null`), but succeeded on `COUNT(*)` and `GROUP BY`.
*   This MVI seems sufficient to provide the necessary data points for feedback on tool choice, SQL generation, and final answer quality. 

## Human Review Setup (Step 2 Prep)

*   Configured 6 Human Review fields in the BrainTrust UI (Project Configuration -> Human Review):
    *   `Tool Choice` (Categorical: Pass/Fail)
    *   `Tool Choice Reason` (Text)
    *   `SQL Correctness` (Categorical: Pass/Fail)
    *   `SQL Correctness Reason` (Text)
    *   `Final Answer Quality` (Categorical: Pass/Fail)
    *   `Final Answer Quality Reason` (Text)
*   **Next Action:** Manually review and label the 15 log traces in the BrainTrust UI Logs/Review section using these configured fields. 
*   **Update:** This manual labeling has been completed. The labeled logs have been fetched and saved to `fetched_braintrust_logs.json`.

## Log Retrieval Troubleshooting

Attempted to programmatically retrieve project logs using the Python SDK to potentially use them for creating evaluation datasets or other analysis (related to Step 2).

*   **Goal:** Fetch logs using Project ID (`7b67d69f-2f0d-4102-a5fd-e448681d6627`).
*   **Documentation Found:** The Human Review page (`https://www.braintrust.dev/docs/guides/human-review#filtering-using-feedback`) suggested using `braintrust.projects.logs.fetch("<project_id>", "<query>")`.
*   **Attempts:**
    *   Calling `braintrust.projects.logs.fetch(project_id=PROJECT_ID)` directly.
    *   Calling `braintrust.init(project=PROJECT_NAME)` then `braintrust.projects.logs.fetch(...)`.
    *   Initializing `client = braintrust.Client()` then `client.projects.logs.fetch(...)`. (Failed with `AttributeError: module 'braintrust' has no attribute 'Client'`)
*   **Result:** All attempts involving `projects.logs.fetch` failed with `AttributeError: 'ProjectBuilder' object has no attribute 'logs'`.
*   **Status:** Asked Braintrust support via Discord for clarification on the correct method. Waiting for response. The current state of `src/agent/braintrust/fetch_labeled_logs.py` reflects the latest attempt. 

**Update: Success via BTQL API**

*   Further attempts using the `requests` library directly also failed with various authentication or syntax errors based on documentation snippets.
*   **Success:** Found that using the `/btql` endpoint with `requests` and the correct BTQL syntax works:
    *   **Method:** `POST https://api.braintrust.dev/btql`
    *   **Auth:** `Authorization: Bearer <API_KEY>`
    *   **Body:** `{"query": "select: *\nfrom: project_logs('<PROJECT_ID>')", "fmt": "json"}`
*   Successfully fetched all 56 log/span records for the project and saved them to `src/agent/braintrust/fetched_braintrust_logs.json`.
*   Confirmed that hand labels (e.g., "Final Answer Quality") are present in the `scores` field of the relevant fetched records.
*   The script `src/agent/braintrust/braintrust_test_run_example.py` contains the working code (though needs renaming/refactoring). 

## Dataset Creation via UI

*   The 15 initial test runs were manually labeled in the BrainTrust UI.
*   These labeled log records were then copied into a new BrainTrust Dataset named `SQL-agent-annotated` using the UI's "Add to dataset" feature.
*   **Note:** While this dataset contains the necessary information (original input, agent output, human labels), the structure within the BrainTrust UI and the fetched records might not directly map to the `input`, `expected` (human labels), and `metadata` (agent output) structure needed by `braintrust.Eval`. The evaluation script (`src/agent/braintrust/evaluate.py`) will need to parse the fetched dataset records and correctly extract/remap these fields.

## Evaluation Data Preparation

*   **Challenge:** Significant difficulties were encountered when trying to programmatically fetch the `SQL-agent-annotated` dataset using the BrainTrust SDK (`braintrust.init_dataset`) in a format suitable for `braintrust.Eval`. Specifically, the human review scores and actual agent outputs (tool calls, final answer) were not readily available in the expected structure.
*   **Resolution:** Due to these challenges, the evaluation dataset was manually constructed based on the notebook exploration (`src/agent/braintrust/explore_eval_dataset.ipynb`). The correctly structured data, including inputs, human-labeled scores/reasons, and agent outputs, has been saved to `src/agent/braintrust/eval_cases_prepared.json`.
*   **Next Step:** Develop the evaluation script (`src/agent/braintrust/evaluate.py`) to load data from `eval_cases_prepared.json` and run evaluations using `braintrust.Eval`. Refer to the BrainTrust documentation for guidance: [Writing Evals](https://www.braintrust.dev/docs/guides/evals/write). 