"""
AI-Powered Trade Reconciliation Discrepancy Detector

What this script does, start to finish:
  1. Loads two transaction files (book A and book B) from the data/ folder.
  2. Finds reconciliation "breaks": trades whose amounts differ by more than a
     small tolerance, or trades that exist in only one of the two books.
  3. Sends each break to the Anthropic API, which returns a plain-English
     analysis as JSON: break_type, likely_cause, severity, suggested_action.
  4. Writes everything to output/report.csv.
"""

import json
import os

import anthropic
import pandas as pd
from dotenv import load_dotenv

# load_dotenv() with no argument looks for a .env file starting in the current
# folder. No hardcoded personal path, so the script runs on anyone's machine.
load_dotenv()

# Settings kept in one place so they're easy to find and change.
# The model can be overridden from .env without touching the code.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
TOLERANCE = 0.01  # differences of a cent or less are treated as rounding, not breaks

# The client reads ANTHROPIC_API_KEY from the environment (loaded from .env above).
# timeout: give up on a single call after 30 seconds instead of hanging.
# max_retries: the SDK automatically retries rate limits (429) and server errors (5xx).
client = anthropic.Anthropic(timeout=30.0, max_retries=3)


class ReconciliationReport:
    """Compares two transaction DataFrames and identifies reconciliation breaks."""

    def __init__(self, df_a, df_b, tolerance=TOLERANCE):
        # Fail loudly with a clear message instead of calling exit() inside a class.
        if df_a.empty or df_b.empty:
            raise ValueError("Both input files must contain at least one trade.")
        self.df_a = df_a
        self.df_b = df_b
        self.tolerance = tolerance
        self.breaks = None

    def find_breaks(self):
        """Returns the rows that don't reconcile, with a break_reason column."""
        # Outer merge keeps trades that appear in only one book.
        # suffixes make the columns read amount_a / amount_b instead of _x / _y.
        merged = self.df_a.merge(
            self.df_b, how="outer", on="trade_id", suffixes=("_a", "_b")
        )

        missing_in_a = merged["amount_a"].isna()
        missing_in_b = merged["amount_b"].isna()
        # Compare with a tolerance, never with exact != on floats.
        amount_mismatch = (merged["amount_a"] - merged["amount_b"]).abs() > self.tolerance

        # Rule-based reason, decided by the code before any AI is involved.
        merged["break_reason"] = None
        merged.loc[amount_mismatch, "break_reason"] = "Amount mismatch"
        merged.loc[missing_in_a, "break_reason"] = "Missing in book A"
        merged.loc[missing_in_b, "break_reason"] = "Missing in book B"

        is_break = missing_in_a | missing_in_b | amount_mismatch
        self.breaks = merged[is_break].reset_index(drop=True)
        return self.breaks

    def summary(self):
        """Returns the total absolute difference across trades found in both books."""
        both = self.breaks.dropna(subset=["amount_a", "amount_b"])
        return (both["amount_a"] - both["amount_b"]).abs().sum()

    def export_report(self, df, file_name):
        """Writes a DataFrame to CSV, creating the output folder if needed."""
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        df.to_csv(file_name, index=False)


def call_api(row):
    """Asks the model to analyze one break. Returns a dict, or None if parsing fails."""
    prompt = (
        "You are a financial data analyst reviewing a trade reconciliation break.\n"
        f"trade_id: {row['trade_id']}\n"
        f"amount in book A: {row['amount_a']}\n"
        f"amount in book B: {row['amount_b']}\n"
        f"rule-based reason: {row['break_reason']}\n\n"
        "Respond with ONLY a JSON object with exactly these keys: "
        '"break_type", "likely_cause", "severity" (Low, Medium or High), '
        '"suggested_action". No other text.'
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    # Models sometimes wrap JSON in ```json fences; strip them before parsing.
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Warning: response for {row['trade_id']} was not valid JSON:\n{raw}")
        return None


FALLBACK = {
    "break_type": "Unknown",
    "likely_cause": "AI response could not be parsed",
    "severity": "Unknown",
    "suggested_action": "Manual review required",
}


if __name__ == "__main__":
    book_a = pd.read_csv("data/book_a.csv")
    book_b = pd.read_csv("data/book_b.csv")

    report = ReconciliationReport(book_a, book_b)
    breaks = report.find_breaks()
    print(f"Found {len(breaks)} breaks. Total amount difference: {report.summary():,.2f}")

    # to_dict("records") gives one plain dict per row: simpler than iterrows().
    results = []
    for row in breaks.to_dict("records"):
        print(f"Analyzing break: {row['trade_id']}...")
        results.append(call_api(row) or FALLBACK)

    # Put the AI columns next to the break rows (same order, same length).
    final = pd.concat([breaks, pd.DataFrame(results)], axis=1)
    report.export_report(final, "output/report.csv")
    print("Report exported to output/report.csv")
