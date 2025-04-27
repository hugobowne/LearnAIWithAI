# Transcript Agent Specification (MVP)

This document outlines the specification for the Minimum Viable Product (MVP) of the transcript query agent.

## 1. Personas

The primary users for the MVP are:

**A. Course Instructor:**
*   **Goal:** Quickly recall points, analyze presenter contributions, find topic examples from Workshop 1.
*   **Needs:** Accurate retrieval, filtering by speaker/topic.

**B. Student / Participant:**
*   **Goal:** Review topics, understand points made, clarify concepts from Workshop 1.
*   **Needs:** Easy-to-understand answers, keyword searching, potentially timestamps.

## 2. MVP Scope

*   Focus solely on the data from `workshop1_transcript.db` (Workshop 1).
*   Primary capability: Answer natural language questions using a Text-to-SQL approach via the `query_database` tool.
*   No complex analysis or cross-segment correlation (defer `analyze_data` tool).
*   No persistent conversation history (each query is treated independently for now).

## 3. Scenarios & Synthetic Test Queries

This list represents typical interactions for the target personas and will be used for testing and refinement.

1.  **Scenario (Student):** Recall details about a specific topic.
    *   **Query:** `"Summarize the key points about evaluation driven development."`
2.  **Scenario (Instructor):** Identify participants/roles.
    *   **Query:** `"List the names of the builders in residence mentioned."`
3.  **Scenario (Student):** Check if a specific term was used and when.
    *   **Query:** `"Find mentions of 'non-determinism' and provide timestamps."`
4.  **Scenario (Instructor):** Recall specific speaker introductions.
    *   **Query:** `"What did Stefan Krawczyk say during his introduction?"`
5.  **Scenario (Student):** Identify who discussed a specific company/entity.
    *   **Query:** `"Who mentioned Carvana?"`
6.  **Scenario (Instructor):** Verify if a topic was covered.
    *   **Query:** `"Did the transcript mention monitoring?"`
7.  **Scenario (Student):** Find information within a time range.
    *   **Query:** `"What software tools were discussed before the 15-minute mark?"`
8.  **Scenario (Instructor):** Get basic information about a person mentioned.
    *   **Query:** `"Who is Jeff Pidcock?"`

## 4. Known Limitations / Future Work

*   Handling of speaker names in queries needs refinement (Prompt Engineering).
*   No complex analysis tools (`analyze_data`, `visualize_data`).
*   No conversation history.
*   Only supports Workshop 1 data. 