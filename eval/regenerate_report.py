import json
import pandas as pd
import numpy as np
import os
import sys # To handle potential path issues if run from outside eval/

# --- Configuration ---
# Simpler path handling: Assume script is in eval/ and paths are relative to it.
script_dir = os.path.dirname(os.path.abspath(__file__))
JUDGED_JSON_PATH = os.path.join(script_dir, "test-set-llm-judged.json")
BASELINE_METRICS_PATH = os.path.join(script_dir, "baseline_metrics.json")
OUTPUT_REPORT_PATH = os.path.join(script_dir, "comparison_report.txt")

# Constants needed if we had to recalculate cost (but it should be in JSON)
PROMPT_OVERHEAD_TOKENS = 100
PRICING = {
    "gpt-3.5-turbo-16k": {"input": 0.0005, "output": 0.0015}, # Example pricing / 1k tokens
    "DEFAULT_COMPLETION": {"input": 0.0005, "output": 0.0015}
}
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.00002, # Example pricing / 1k tokens
    "DEFAULT_EMBEDDING": 0.00002
}

# --- Helper Functions (Copied/Adapted from evaluate_system.py) ---
def calculate_stats(series):
    """Calculates average, median, and P95 for a numeric series."""
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if numeric_series.empty:
        return np.nan, np.nan, np.nan
    avg = numeric_series.mean()
    median = numeric_series.median() # Median is not used in report, but keep for consistency
    p95 = numeric_series.quantile(0.95)
    return avg, median, p95

def calculate_estimated_cost(row):
    """Estimates the cost for a single interaction row."""
    # This shouldn't be needed as cost is pre-calculated in the judged JSON,
    # but kept here for completeness if needed later.
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
        # print(f"Warning: Could not calculate cost for a row: {e}") # Optional warning
        return np.nan

def calculate_pass_rate(df, verdict_col='llm_judge_verdict'):
    """Calculates the PASS rate from a specific verdict column."""
    valid_verdicts = df[verdict_col].dropna()
    num_judged = len(valid_verdicts)
    if num_judged == 0:
        return None, 0
    pass_count = (valid_verdicts == 'PASS').sum()
    pass_rate = (pass_count / num_judged) * 100
    return pass_rate, num_judged

def format_change(current, baseline):
    """Formats the change between current and baseline metrics."""
    if pd.isna(current) or pd.isna(baseline) or baseline == 0:
        return "N/A"

    change_abs = current - baseline
    # Handle cases where baseline might be zero for percentage calculation
    change_pct = ((current - baseline) / baseline) * 100 if baseline != 0 else np.inf if current != 0 else 0


    # Choose format based on magnitude and context (rates vs counts/latency)
    # Simplification: use percentage points for pass_rate, cost. Use % for others.
    # A more robust check might be needed depending on metric types.
    # Assume pass_rate and cost are typically < 1 for pp formatting.
    if abs(baseline) < 1 and baseline != 0 : # Crude check for rates/costs likely < 1
         change_pp = change_abs * 100 # Percentage points
         # Ensure change_pp is formatted correctly even if change_abs is very small
         return f"{change_pp:+.1f}pp"
    elif baseline != 0: # For latency, tokens etc.
        # Format absolute change based on magnitude
        abs_fmt = f"{change_abs:+.2f}" if abs(change_abs) >= 0.01 else f"{change_abs:+.{max(2, -int(np.floor(np.log10(abs(change_abs)))))}f}" if change_abs != 0 else "+0.00"
        return f"{abs_fmt} ({change_pct:+.1f}%)"
    else: # Handle baseline is 0 cases
        return f"{current:+.2f} (New)" if current != 0 else "No Change"


def generate_comparison_report(current_metrics, baseline_metrics):
    """Generates the comparison report string."""
    report_lines = []
    report_lines.append("--- Evaluation Comparison Report ---")

    if not baseline_metrics:
        report_lines.append("Baseline metrics not available for comparison.")
        return "\n".join(report_lines)

    baseline_name = baseline_metrics.get('baseline_version_name', 'Baseline')
    current_judged = current_metrics.get('num_judged', 'N/A')
    baseline_rated = baseline_metrics.get('based_on_num_rated', 'N/A')

    report_lines.append(f"Current Run vs. {baseline_name}")
    report_lines.append(f"Compared {current_judged} judged results against {baseline_rated} baseline results.")
    report_lines.append("") # Newline

    # --- Quality ---
    report_lines.append("--- Quality (PASS Rate %) ---")
    c_pass = current_metrics.get('quality', {}).get('pass_rate_pct')
    b_pass = baseline_metrics.get('quality', {}).get('pass_rate_pct')
    report_lines.append(f"Current:          {c_pass:.1f}%" if c_pass is not None else "Current:          N/A")
    report_lines.append(f"Baseline:         {b_pass:.1f}%" if b_pass is not None else "Baseline:         N/A")
    report_lines.append(f"Change:           {format_change(c_pass / 100.0 if c_pass is not None else np.nan, b_pass / 100.0 if b_pass is not None else np.nan)}") # Pass rate change in pp
    report_lines.append("")

    # --- Latency ---
    report_lines.append("--- Latency (seconds) ---")
    c_lat_avg = current_metrics.get('latency_seconds', {}).get('average')
    b_lat_avg = baseline_metrics.get('latency_seconds', {}).get('average')
    c_lat_p95 = current_metrics.get('latency_seconds', {}).get('p95')
    b_lat_p95 = baseline_metrics.get('latency_seconds', {}).get('p95')
    report_lines.append(f"Avg Current:      {c_lat_avg:.2f}s" if c_lat_avg is not None else "Avg Current:      N/A")
    report_lines.append(f"Avg Baseline:     {b_lat_avg:.2f}s" if b_lat_avg is not None else "Avg Baseline:     N/A")
    report_lines.append(f"Avg Change:       {format_change(c_lat_avg, b_lat_avg)}")
    report_lines.append("---")
    report_lines.append(f"P95 Current:      {c_lat_p95:.2f}s" if c_lat_p95 is not None else "P95 Current:      N/A")
    report_lines.append(f"P95 Baseline:     {b_lat_p95:.2f}s" if b_lat_p95 is not None else "P95 Baseline:     N/A")
    report_lines.append(f"P95 Change:       {format_change(c_lat_p95, b_lat_p95)}")
    report_lines.append("")

    # --- Tokens ---
    report_lines.append("--- Tokens (total per query) ---")
    c_tok_avg = current_metrics.get('tokens_total', {}).get('average')
    b_tok_avg = baseline_metrics.get('tokens_total', {}).get('average')
    report_lines.append(f"Avg Current:      {c_tok_avg:.0f}" if c_tok_avg is not None else "Avg Current:      N/A")
    report_lines.append(f"Avg Baseline:     {b_tok_avg:.0f}" if b_tok_avg is not None else "Avg Baseline:     N/A")
    report_lines.append(f"Avg Change:       {format_change(c_tok_avg, b_tok_avg)}")
    report_lines.append("")

    # --- Cost ---
    report_lines.append("--- Estimated Cost (USD per query) ---")
    c_cost_avg = current_metrics.get('estimated_cost_usd', {}).get('average')
    b_cost_avg = baseline_metrics.get('estimated_cost_usd', {}).get('average')
    report_lines.append(f"Avg Current:      ${c_cost_avg:.6f}" if c_cost_avg is not None else "Avg Current:      N/A")
    report_lines.append(f"Avg Baseline:     ${b_cost_avg:.6f}" if b_cost_avg is not None else "Avg Baseline:     N/A")
    report_lines.append(f"Avg Change:       {format_change(c_cost_avg, b_cost_avg)}") # Cost change in pp
    report_lines.append("")
    report_lines.append("------------------------------------")

    return "\n".join(report_lines)

# --- Main Logic ---
def main():
    print(f"--- Regenerating Comparison Report ---")
    print(f"Loading judged results from: {JUDGED_JSON_PATH}")
    print(f"Loading baseline metrics from: {BASELINE_METRICS_PATH}")

    # Load judged results
    try:
        # Read JSON into a list of dicts first
        with open(JUDGED_JSON_PATH, 'r', encoding='utf-8') as f:
            judged_data_list = json.load(f)
        # Convert to DataFrame
        results_df = pd.DataFrame(judged_data_list)
        print(f"Successfully loaded {len(results_df)} judged records.")
    except FileNotFoundError:
        print(f"Error: Judged results file not found at {JUDGED_JSON_PATH}")
        return
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {JUDGED_JSON_PATH}. Is the file valid?")
         return
    except Exception as e:
        print(f"Error loading judged results JSON: {e}")
        return

    # Load baseline metrics
    try:
        with open(BASELINE_METRICS_PATH, 'r') as f:
            baseline_metrics = json.load(f)
        print("Baseline metrics loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Baseline metrics file not found at {BASELINE_METRICS_PATH}")
        baseline_metrics = None
    except Exception as e:
        print(f"Error loading baseline metrics JSON: {e}")
        baseline_metrics = None

    # --- Calculate Metrics for the Judged Run ---
    print("\nCalculating metrics from judged data...")
    current_metrics = {}

    # Calculate Automated PASS Rate
    pass_rate, num_judged = calculate_pass_rate(results_df, verdict_col='llm_judge_verdict')
    current_metrics['num_judged'] = num_judged
    current_metrics['quality'] = {"pass_rate_pct": pass_rate if num_judged > 0 else None}

    # Calculate performance/cost stats
    avg_latency, _, p95_latency = calculate_stats(results_df.get('latency')) # Use .get for safety
    current_metrics['latency_seconds'] = {"average": avg_latency, "p95": p95_latency}

    avg_tokens, _, _ = calculate_stats(results_df.get('total_tokens'))
    current_metrics['tokens_total'] = {"average": avg_tokens}

    # Cost should already be calculated in the JSON file
    if 'estimated_cost_usd' not in results_df.columns:
         print("Warning: 'estimated_cost_usd' column not found in judged JSON. Calculating it now.")
         # Ensure 'model' column exists or handle missing values gracefully
         results_df['estimated_cost_usd'] = results_df.apply(lambda row: calculate_estimated_cost(row) if 'model' in row else np.nan, axis=1)

    avg_cost, _, _ = calculate_stats(results_df.get('estimated_cost_usd'))
    current_metrics['estimated_cost_usd'] = {"average": avg_cost}

    # --- Generate and Print/Save Report ---
    print("\nGenerating comparison report...")
    report_content = generate_comparison_report(current_metrics, baseline_metrics)

    print("\n" + report_content) # Print to console

    # Save report to file
    try:
        with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"\nSuccessfully saved comparison report to {OUTPUT_REPORT_PATH}")
    except Exception as e:
        print(f"\nError saving report to file: {e}")


if __name__ == "__main__":
    main() 