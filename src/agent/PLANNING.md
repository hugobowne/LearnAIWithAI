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
    *   Create a script (`evaluate.py` perhaps?) that takes the labeled test set.
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