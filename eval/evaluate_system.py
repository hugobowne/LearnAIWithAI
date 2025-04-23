"""
LLM-as-Judge Evaluation Harness

Purpose:
--------
This script evaluates a given set of pre-generated system outputs using an 
LLM-as-Judge and compares the results to a pre-calculated baseline. 
It forms the core of the automated evaluation process for iterating on the Q&A system.

Methodology:
------------
1.  **Decoupled Design:** Takes a CSV file containing pre-generated results 
    (question, response, latency, tokens, etc.) as input via a command-line 
    argument. It does *not* run the Q&A system itself, allowing flexibility 
    in evaluating different system versions or architectures.
2.  **LLM-as-Judge:** For each input record, it calls a powerful LLM 
    (e.g., gpt-4-turbo) configured as a 'judge'.
    *   The judge receives the question and the generated response.
    *   It uses a prompt with instructions and few-shot examples to evaluate 
      the response based on helpfulness, relevance, and directness.
    *   It outputs a verdict ('PASS'/'FAIL') and a brief reason in JSON format.
3.  **Metrics Calculation:** Calculates metrics for the evaluated run:
    *   Automated PASS Rate (%) based on judge verdicts.
    *   Performance stats (Avg/P95 Latency) read from the input CSV.
    *   Cost/Usage stats (Avg Tokens, Avg Estimated Cost) read/calculated 
      from the input CSV.
4.  **Comparison (TODO):** The script is intended to load baseline metrics and 
    present a comparison report (this part is not yet implemented).
5.  **Output:** Prints metrics for the current run. Detailed judge outputs 
    (verdict + reason per item) are added as columns to the DataFrame 
    (and could be saved to an output file). 

Evaluation Philosophy & Limitations (MVE Context):
-------------------------------------------------
*   **Focus on End-to-End Quality:** The LLM judge primarily assesses the quality 
    of the final response from an end-user perspective (helpful, relevant?).
*   **No Groundedness Check:** The current judge **does not perform 
    faithfulness/groundedness checks** as it does not receive the retrieved 
    context. It may PASS plausible but factually incorrect responses.
*   **Calibration Needed:** The judge's alignment with human judgment should be 
    checked periodically by comparing its verdicts to human evaluations on a 
    subset of the data.

Future Enhancements:
--------------------
*   Implement the comparison logic against baseline metrics.
*   Add optional saving of detailed results (including judge reasons).
*   Incorporate deterministic checks (e.g., regex) for specific factual questions.
*   Add faithfulness evaluation by providing context to the judge (or a separate judge).
"""

import pandas as pd
import numpy as np
import os
import argparse
import openai # We'll need this later for the LLM judge
import json # For parsing judge output
from dotenv import load_dotenv
import time # For potential retries/rate limiting

# --- Configuration ---
BASELINE_METRICS_PATH = os.path.join(os.path.dirname(__file__), "baseline_metrics.json")
JUDGE_MODEL = "gpt-4-turbo" # Define Judge Model
PROMPT_OVERHEAD_TOKENS = 100 # Assumed tokens for system prompt, question text etc.

# --- Pricing (Approximate - Update as needed!) ---
# Prices per 1K tokens in USD
PRICING = {
    "gpt-3.5-turbo-16k": { # Assuming same as gpt-3.5-turbo-0125
        "input": 0.0005, 
        "output": 0.0015
    },
    "gpt-4-turbo": {
        "input": 0.01,
        "output": 0.03
    },
    "DEFAULT_COMPLETION": { # Fallback if model not listed
        "input": 0.001, 
        "output": 0.002
    }
}
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.00002,
    "text-embedding-ada-002": 0.0001,
    "DEFAULT_EMBEDDING": 0.0001 # Fallback
}

# --- Few-Shot Examples --- (Selected from test-set.csv)
FEW_SHOT_EXAMPLES = """
--- Examples ---

Question: What's the recipe for fettuccini alfredo?
Response: I'm sorry, but I couldn't find any information about the recipe for fettuccini alfredo in the provided workshop transcript sections.
Evaluation JSON:
{
  "verdict": "PASS",
  "reason": "The response correctly identifies that the information is not in the likely context and avoids hallucination."
}

Question: who are the instructors?
Response: The instructors mentioned in the workshop transcript sections are Hugo Bowne-Anderson, Stefan Krawczyk, Jeff Pidcock, Nathan Danielson, and William Horton.
Evaluation JSON:
{
  "verdict": "FAIL",
  "reason": "The response includes builders-in-residence (Jeff, Nathan, William) as instructors, which is incorrect based on their defined roles."
}

Question: What's Different about Evaluating LLM Apps?
Response: The workshop transcript mentions that evaluating LLM (Language Model) apps is different from traditional software development because LLM-powered software brings in a lot of data from the real world and its behavior is non-deterministic... [rest of summary]
Evaluation JSON:
{
  "verdict": "PASS",
  "reason": "The response accurately summarizes the key differences mentioned in the likely context regarding evaluation-driven development and non-determinism."
}

--- End Examples ---
"""

# --- Judge Prompt Template ---
JUDGE_PROMPT_TEMPLATE = f"""You are an impartial evaluator assessing the quality of answers provided by an AI assistant to questions based on a workshop transcript.
The user who asked the question has provided feedback on similar answers previously, focusing on whether the answer was helpful, relevant, and directly addressed their question.
Your goal is to provide a 'PASS' or 'FAIL' verdict based on these criteria. Do NOT evaluate based on information external to the question-answer pair.
Output your evaluation ONLY as a JSON object with two keys:
1.  "verdict": Must be either "PASS" or "FAIL".
2.  "reason": A brief explanation (1-2 sentences) justifying your verdict based on helpfulness, relevance, and directness.

{FEW_SHOT_EXAMPLES}

--- Task ---
Evaluate the following response based on the question:

Question: {{QUESTION}}
Response: {{RESPONSE}}

Evaluation JSON:
"""

# --- Helper Functions (Copied from run_baseline_eval.py) ---
def calculate_stats(series):
    """Calculates average, median, and P95 for a numeric series."""
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if numeric_series.empty:
        return np.nan, np.nan, np.nan
    avg = numeric_series.mean()
    median = numeric_series.median()
    p95 = numeric_series.quantile(0.95)
    return avg, median, p95

def calculate_estimated_cost(row):
    """Estimates the cost for a single interaction row."""
    try:
        model = row.get('model', "DEFAULT_COMPLETION")
        completion_pricing = PRICING.get(model, PRICING["DEFAULT_COMPLETION"])
        embedding_model_price = EMBEDDING_PRICING.get("text-embedding-3-small", EMBEDDING_PRICING["DEFAULT_EMBEDDING"])

        context_tokens = pd.to_numeric(row.get('context_tokens', 0), errors='coerce')
        completion_tokens = pd.to_numeric(row.get('completion_tokens', 0), errors='coerce')
        embedding_tokens = pd.to_numeric(row.get('embedding_tokens', 0), errors='coerce')
        
        context_tokens = 0 if pd.isna(context_tokens) else context_tokens
        completion_tokens = 0 if pd.isna(completion_tokens) else completion_tokens
        embedding_tokens = 0 if pd.isna(embedding_tokens) else embedding_tokens
        
        input_tokens_est = context_tokens + PROMPT_OVERHEAD_TOKENS
        
        embedding_cost = (embedding_tokens / 1000) * embedding_model_price
        completion_cost = (input_tokens_est / 1000) * completion_pricing["input"] + \
                          (completion_tokens / 1000) * completion_pricing["output"]
                          
        total_cost = embedding_cost + completion_cost
        return total_cost
    except Exception as e:
        return np.nan

def get_llm_judge_verdict(client, question, response):
    """Calls the LLM judge to get a verdict and reason for a given Q&A pair."""
    prompt = JUDGE_PROMPT_TEMPLATE.replace("{{QUESTION}}", question).replace("{{RESPONSE}}", response)
    
    try:
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, 
            max_tokens=150, # Enough for JSON verdict + reason
            response_format={ "type": "json_object" } # Request JSON output if model supports it
        )
        
        judge_response_text = completion.choices[0].message.content
        
        # Parse the JSON response
        try:
            evaluation = json.loads(judge_response_text)
            verdict = evaluation.get("verdict")
            reason = evaluation.get("reason")
            
            # Validate verdict
            if verdict in ["PASS", "FAIL"]:
                return verdict, reason if reason else "No reason provided."
            else:
                print(f"Warning: Invalid verdict '{verdict}' received from judge.")
                return None, f"Invalid verdict: {verdict}"
                
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON judge response: {judge_response_text}")
            return None, f"Parsing Error: {judge_response_text}"
        except Exception as e:
            print(f"Warning: Error processing judge response JSON: {e}")
            return None, f"Processing Error: {judge_response_text}"

    except Exception as e:
        print(f"Error calling LLM Judge API: {e}")
        # Basic retry logic (optional, can be enhanced)
        # time.sleep(1)
        # try: ... second attempt ...
        return None, f"API Error: {e}"

# --- Main Execution ---
def main(input_csv_path):
    print(f"--- Running Evaluation Harness ---")
    print(f"Input results file: {input_csv_path}")
    
    # Load the results dataset
    try:
        results_df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded {len(results_df)} records from {input_csv_path}")

        # --- Temporary modification for testing ---
        # results_df = results_df.head(2) # Process only the first 2 rows
        # print(f"\n*** Running in test mode on first 2 rows only! ***\n")
        # --- End Temporary modification ---

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        print(f"Error loading input CSV: {e}")
        return

    # --- Initialize OpenAI Client --- (Moved up)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file. Cannot run LLM Judge.")
        return
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return
    # --- End Client Init ---
    
    # --- Run LLM Judge --- #
    print(f"\nRunning LLM Judge ({JUDGE_MODEL}) on {len(results_df)} records...")
    judge_results = results_df.apply(
        lambda row: pd.Series(get_llm_judge_verdict(client, row['question'], row['response']), index=['llm_judge_verdict', 'llm_judge_reason']),
        axis=1
    )
    results_df = pd.concat([results_df, judge_results], axis=1)
    print("LLM Judge evaluation complete.")
    
    # --- Save Judged Results to JSON --- 
    base_name = os.path.basename(input_csv_path)
    name, _ = os.path.splitext(base_name)
    output_filename = f"{name}-llm-judged.json" # JSON extension
    output_path = os.path.join(os.path.dirname(input_csv_path), output_filename)
    
    try:
        # Convert DataFrame to list of dicts for JSON, handle potential NaNs
        results_list = results_df.replace({np.nan: None}).to_dict(orient='records')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=4) # Use indent for readability
        print(f"\nSuccessfully saved judged results to {output_path}")
    except Exception as e:
        print(f"\nError saving judged results to JSON: {e}")
    # --- End Save Judged Results ---
    
    # Count successful judgements
    successful_judgements = results_df['llm_judge_verdict'].notna().sum()
    print(f"Successfully judged {successful_judgements} out of {len(results_df)} records.")

    # --- Calculate Metrics for this Run --- #
    print("\nCalculating metrics for this run...")
    current_metrics = {}
    current_metrics['num_judged'] = successful_judgements
    
    # Calculate Automated PASS Rate
    pass_rate, num_judged = calculate_pass_rate(results_df, verdict_col='llm_judge_verdict') 
    current_metrics['quality'] = {"pass_rate_pct": pass_rate if num_judged > 0 else None}

    # Calculate performance/cost stats from input file
    avg_latency, _, p95_latency = calculate_stats(results_df['latency'])
    current_metrics['latency_seconds'] = {"average": avg_latency, "p95": p95_latency}
    
    avg_tokens, _, _ = calculate_stats(results_df['total_tokens'])
    current_metrics['tokens_total'] = {"average": avg_tokens}
    
    results_df['estimated_cost_usd'] = results_df.apply(calculate_estimated_cost, axis=1)
    avg_cost, _, _ = calculate_stats(results_df['estimated_cost_usd'])
    current_metrics['estimated_cost_usd'] = {"average": avg_cost}

    # --- Load Baseline Metrics --- 
    print(f"\nLoading baseline metrics from {BASELINE_METRICS_PATH}...")
    try:
        with open(BASELINE_METRICS_PATH, 'r') as f:
            baseline_metrics = json.load(f)
        print("Baseline metrics loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Baseline metrics file not found at {BASELINE_METRICS_PATH}")
        print("Cannot generate comparison report. Please run run_baseline_eval.py first.")
        baseline_metrics = None # Set baseline to None if not found
    except Exception as e:
        print(f"Error loading baseline metrics JSON: {e}")
        baseline_metrics = None # Set baseline to None on error
    
    # --- Print Comparison Report --- 
    if baseline_metrics:
        print_comparison_report(current_metrics, baseline_metrics)
    else:
        # Optionally print just the current metrics if baseline isn't available
        print("\n--- Metrics for Current Run (Automated Eval) --- ")
        print(f"Evaluated {len(results_df)} records, successfully judged {num_judged}.")
        print(f"Automated PASS Rate: {current_metrics['quality'].get('pass_rate_pct', 'N/A'):.1f}%")
        print(f"Average Latency: {current_metrics['latency_seconds'].get('average', 'N/A'):.2f}s")
        print(f"P95 Latency: {current_metrics['latency_seconds'].get('p95', 'N/A'):.2f}s")
        print(f"Average Tokens: {current_metrics['tokens_total'].get('average', 'N/A'):.0f}")
        print(f"Average Est. Cost: ${current_metrics['estimated_cost_usd'].get('average', 'N/A'):.6f}")
        print("\n------------------------------------")

def calculate_pass_rate(df, verdict_col='feedback_rating'):
    """Calculates the PASS rate from the specified verdict column."""
    rated_df = df[df[verdict_col].isin(['PASS', 'FAIL'])].copy()
    if rated_df.empty:
        return 0.0, 0
    pass_count = (rated_df[verdict_col] == 'PASS').sum()
    total_rated = len(rated_df)
    pass_rate = (pass_count / total_rated) * 100 if total_rated > 0 else 0
    return pass_rate, total_rated

def format_change(current, baseline):
    """Formats the change between current and baseline metrics."""
    if baseline is None or current is None or np.isnan(baseline) or np.isnan(current):
        return "N/A"
    change = current - baseline
    pct_change = ((current - baseline) / baseline) * 100 if baseline != 0 else 0
    
    # Format based on type (e.g., percentage points for rates, absolute for others)
    if abs(baseline) <= 100 and abs(current) <= 100: # Heuristic for percentages
         change_str = f"{change:+.1f}pp"
    elif isinstance(baseline, float) and abs(baseline) < 0.1: # Heuristic for costs
         change_str = f"{change:+.6f}"
    elif isinstance(baseline, float):
         change_str = f"{change:+.2f}"
    else:
         change_str = f"{change:+.0f}"

    # Add percentage change for non-percentage values
    if not (abs(baseline) <= 100 and abs(current) <= 100):
       change_str += f" ({pct_change:+.1f}%)"
       
    return change_str

def print_comparison_report(current_metrics, baseline_metrics):
    """Prints a formatted comparison report."""
    print("\n--- Evaluation Comparison Report --- ")
    print(f"Current Run vs. {baseline_metrics.get('baseline_version_name', 'Unknown Baseline')}")
    print(f"Compared {current_metrics['num_judged']} judged results against {baseline_metrics.get('based_on_num_rated', 'N/A')} baseline results.")

    # --- Quality ---
    print("\n--- Quality (PASS Rate %) --- ")
    bl_pass = baseline_metrics.get('quality', {}).get('pass_rate_pct')
    cr_pass = current_metrics['quality'].get('pass_rate_pct')
    print(f"Current:          {cr_pass:.1f}%" if cr_pass is not None else "Current:          N/A")
    print(f"Baseline:         {bl_pass:.1f}%" if bl_pass is not None else "Baseline:         N/A")
    print(f"Change:           {format_change(cr_pass, bl_pass)}")

    # --- Latency ---
    print("\n--- Latency (seconds) --- ")
    bl_avg_lat = baseline_metrics.get('latency_seconds', {}).get('average')
    cr_avg_lat = current_metrics['latency_seconds'].get('average')
    print(f"Avg Current:      {cr_avg_lat:.2f}s" if cr_avg_lat is not None else "Avg Current:      N/A")
    print(f"Avg Baseline:     {bl_avg_lat:.2f}s" if bl_avg_lat is not None else "Avg Baseline:     N/A")
    print(f"Avg Change:       {format_change(cr_avg_lat, bl_avg_lat)}")
    print("---")
    bl_p95_lat = baseline_metrics.get('latency_seconds', {}).get('p95')
    cr_p95_lat = current_metrics['latency_seconds'].get('p95')
    print(f"P95 Current:      {cr_p95_lat:.2f}s" if cr_p95_lat is not None else "P95 Current:      N/A")
    print(f"P95 Baseline:     {bl_p95_lat:.2f}s" if bl_p95_lat is not None else "P95 Baseline:     N/A")
    print(f"P95 Change:       {format_change(cr_p95_lat, bl_p95_lat)}")

    # --- Tokens ---
    print("\n--- Tokens (total per query) --- ")
    bl_avg_tok = baseline_metrics.get('tokens_total', {}).get('average')
    cr_avg_tok = current_metrics['tokens_total'].get('average')
    print(f"Avg Current:      {cr_avg_tok:.0f}" if cr_avg_tok is not None else "Avg Current:      N/A")
    print(f"Avg Baseline:     {bl_avg_tok:.0f}" if bl_avg_tok is not None else "Avg Baseline:     N/A")
    print(f"Avg Change:       {format_change(cr_avg_tok, bl_avg_tok)}")

    # --- Cost ---
    print("\n--- Estimated Cost (USD per query) --- ")
    bl_avg_cost = baseline_metrics.get('estimated_cost_usd', {}).get('average')
    cr_avg_cost = current_metrics['estimated_cost_usd'].get('average')
    print(f"Avg Current:      ${cr_avg_cost:.6f}" if cr_avg_cost is not None else "Avg Current:      N/A")
    print(f"Avg Baseline:     ${bl_avg_cost:.6f}" if bl_avg_cost is not None else "Avg Baseline:     N/A")
    print(f"Avg Change:       {format_change(cr_avg_cost, bl_avg_cost)}")

    print("\n------------------------------------")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate system results using an LLM-as-Judge and compare to baseline.")
    parser.add_argument("input_csv", help="Path to the CSV file containing the system results (question, response, etc.)")
    args = parser.parse_args()
    
    main(args.input_csv) 