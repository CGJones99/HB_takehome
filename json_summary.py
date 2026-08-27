"""Final JSON summary: per-category counts plus the highest-hit-count ticket in each."""

import json

import anthropic
import pandas as pd

from merge_summary import build_final_table, load_api_key_or_exit, load_csv_or_exit

CATEGORIES = ["refund_return", "shipping", "product_question", "other", "human_review"]


def ticket_number(ticket_id: str) -> int:
    return int(ticket_id.rsplit("-", 1)[1])


def top_hit_ticket(df: pd.DataFrame, category: str) -> "dict | None":
    subset = df[df["classification"] == category]
    if subset.empty:
        return None

    best = min(
        subset.itertuples(),
        key=lambda row: (-(0 if pd.isna(row.hit_count) else row.hit_count), ticket_number(row.ticket_id)),
    )
    return {
        "classification": category,
        "ticket_id": best.ticket_id,
        "subject": None if pd.isna(best.subject) else best.subject,
        "body": None if pd.isna(best.body) else best.body,
    }


def main() -> None:
    api_key = load_api_key_or_exit()
    df = load_csv_or_exit()
    client = anthropic.Anthropic(api_key=api_key)

    df = build_final_table(df, client)

    counts = {category: int((df["classification"] == category).sum()) for category in CATEGORIES}
    summary = {
        "counts": counts,
        "total": sum(counts.values()),
        "top_hits": {category: top_hit_ticket(df, category) for category in CATEGORIES},
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
