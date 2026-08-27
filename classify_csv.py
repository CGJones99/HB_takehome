"""Deterministic keyword classification pass over customer_service_emails.csv."""

import sys
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "customer_service_emails.csv"

KEYWORDS = {
    "refund_return": ["refund", "return", "exchange", "credit", "cancel"],
    "shipping": ["tracking", "delay", "delivery", "address", "shipping cost", "shipping method"],
    "product_question": ["size", "sizing", "material", "care", "fit", "spec", "availability"],
}

CLASSIFICATIONS = ["refund_return", "shipping", "product_question", "other", "llm_review"]


def count_hits(text: str) -> dict[str, int]:
    text = text.lower()
    return {
        category: sum(text.count(kw) for kw in keywords)
        for category, keywords in KEYWORDS.items()
    }


def classify(subject: object, body: object) -> tuple[str, "int | None"]:
    subject = "" if pd.isna(subject) else str(subject)
    body = "" if pd.isna(body) else str(body)
    combined = f"{subject} {body}".strip()

    if not combined:
        return "other", None

    hits = count_hits(combined)
    hit_categories = [c for c, n in hits.items() if n > 0]

    if len(hit_categories) != 1:
        return "llm_review", None

    category = hit_categories[0]
    return category, hits[category]


def main() -> None:
    if not CSV_PATH.exists():
        print(f"Error: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {CSV_PATH} is empty")
        sys.exit(1)

    results = df.apply(lambda row: classify(row.get("subject"), row.get("body")), axis=1)
    df["classification"] = [r[0] for r in results]
    df["hit_count"] = [r[1] for r in results]

    print(f"{'ticket_id':<12} {'classification':<18} {'hit_count'}")
    for _, row in df.iterrows():
        hit_count = "" if pd.isna(row["hit_count"]) else int(row["hit_count"])
        print(f"{row['ticket_id']:<12} {row['classification']:<18} {hit_count}")

    counts = df["classification"].value_counts()
    total = len(df)

    print("\nSummary:")
    summary_total = 0
    for category in CLASSIFICATIONS:
        n = int(counts.get(category, 0))
        summary_total += n
        print(f"  {category:<18} {n}")
    print(f"  {'total':<18} {summary_total}")

    if summary_total == total:
        print(f"\nSum of classifications: {summary_total} / Total rows: {total} - MATCH")
    else:
        print(f"\nSum of classifications: {summary_total} / Total rows: {total} - MISMATCH")


if __name__ == "__main__":
    main()
