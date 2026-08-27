"""Merge step-2 keyword classification with step-3 LLM classification into one table.

No new LLM prompting/schema here - this reuses llm_classify.py's existing
classify_row() to fill in the llm_review rows, then applies two deterministic
rules: attach the LLM's reasoning as a comment, and override to human_review
wherever the LLM reported low confidence.
"""

import os
import sys

import anthropic
import pandas as pd
from dotenv import load_dotenv

from classify_csv import CSV_PATH, classify
from llm_classify import classify_row

CLASSIFICATIONS = ["refund_return", "shipping", "product_question", "other", "llm_review", "human_review"]


def build_final_table(df: pd.DataFrame, client: anthropic.Anthropic) -> pd.DataFrame:
    """Return df with classification/hit_count/llm_comment columns filled in."""
    step2 = df.apply(lambda row: classify(row.get("subject"), row.get("body")), axis=1)
    df["classification"] = [r[0] for r in step2]
    df["hit_count"] = [r[1] for r in step2]
    df["llm_comment"] = None

    for idx, row in df[df["classification"] == "llm_review"].iterrows():
        ticket_id = row["ticket_id"]
        try:
            result = classify_row(client, row.get("subject"), row.get("body"))
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError, ValueError) as e:
            print(f"Error: {ticket_id} - classification failed: {e}")
            df.at[idx, "classification"] = "ERROR"
            df.at[idx, "llm_comment"] = str(e)
            continue

        classification = "human_review" if result.confidence == "low" else result.classification
        df.at[idx, "classification"] = classification
        df.at[idx, "llm_comment"] = result.reasoning

    return df


def load_csv_or_exit() -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        sys.exit(1)
    try:
        return pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {CSV_PATH} is empty")
        sys.exit(1)


def load_api_key_or_exit() -> str:
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("Error: API_KEY not set in .env")
        sys.exit(1)
    return api_key


def main() -> None:
    api_key = load_api_key_or_exit()
    df = load_csv_or_exit()
    client = anthropic.Anthropic(api_key=api_key)

    df = build_final_table(df, client)
    out = df[["ticket_id", "classification", "hit_count", "llm_comment"]]

    print(f"{'ticket_id':<12} {'classification':<18} {'hit_count':<10} llm_comment")
    for _, row in out.iterrows():
        hit_count = "" if pd.isna(row["hit_count"]) else int(row["hit_count"])
        comment = "" if pd.isna(row["llm_comment"]) else row["llm_comment"]
        print(f"{row['ticket_id']:<12} {row['classification']:<18} {str(hit_count):<10} {comment}")

    counts = out["classification"].value_counts()
    total = len(out)

    print("\nSummary:")
    summary_total = 0
    for category in CLASSIFICATIONS:
        n = int(counts.get(category, 0))
        summary_total += n
        print(f"  {category:<18} {n}")
    errors = int(counts.get("ERROR", 0))
    summary_total += errors
    if errors:
        print(f"  {'ERROR':<18} {errors}")
    print(f"  {'total':<18} {summary_total}")

    if summary_total == total:
        print(f"\nSum of classifications: {summary_total} / Total rows: {total} - MATCH")
    else:
        print(f"\nSum of classifications: {summary_total} / Total rows: {total} - MISMATCH")


if __name__ == "__main__":
    main()
