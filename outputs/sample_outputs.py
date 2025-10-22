#!/usr/bin/env python3
"""
sample_models_csv.py

Inputs: 3 CSV files (each must contain columns for current_action, predicted_action, and ONE model-reply column),
        and a number_of_samples.
Output: sampled.csv (unless overridden with --output)

The script:
- Reads the three input CSVs.
- Identifies the key columns: current_action and predicted_action (case-insensitive, underscores/spaces allowed).
- Detects the single "model reply" column in each file and infers the model name from its header.
- Inner-joins on (current_action, predicted_action) so only rows present in ALL THREE files survive.
- Randomly samples number_of_samples rows (or fewer if not enough).
- Writes an output file with columns:
  current_action, predicted_action,
  {model1}_reply, {model1}_score,
  {model2}_reply, {model2}_score,
  {model3}_reply, {model3}_score

Score columns are left empty for later manual filling.
"""

import argparse
import os
import re
from typing import List, Tuple
import pandas as pd


# --- Helpers -----------------------------------------------------------------

KEY_CANDIDATES_CURRENT = {"current_action", "current action"}
KEY_CANDIDATES_PRED = {"predicted_action", "predicted action"}

def _norm(s: str) -> str:
    """Normalize header for matching (lower, strip, collapse spaces -> underscore)."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s

def _find_key_col(cols: List[str], candidates: set) -> str:
    """Find a column in cols whose normalized name matches candidates."""
    for c in cols:
        if _norm(c) in candidates:
            return c
    raise ValueError(f"Could not find any of {sorted(candidates)} among columns: {list(cols)}")

def _infer_model_from_reply_col(reply_col: str) -> str:
    """
    Infer a model name from the reply column header by stripping common suffixes like
    *_reply, *_response, *_output, *_answer (case-insensitive, underscores/spaces/dashes tolerated).
    """
    raw = _norm(reply_col)
    # Remove common suffixes
    raw = re.sub(r"(_|-|\s)?(reply|response|output|answer|prediction)$", "", raw, flags=re.IGNORECASE)
    # Clean up multiple underscores/dashes/spaces
    raw = re.sub(r"[_\s\-]+", "_", raw).strip("_")
    return raw or reply_col  # fallback to original if everything stripped

def _ensure_unique(names: List[str]) -> List[str]:
    """Ensure names are unique by appending _2, _3, ... if needed."""
    seen = {}
    unique = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 1
            unique.append(base)
        else:
            seen[base] += 1
            unique.append(f"{base}_{seen[base]}")
    return unique

def _choose_reply_col(all_cols: List[str], current_col: str, pred_col: str) -> str:
    """Pick the model's reply column. Prefer names containing 'reply'/'response'/'output' etc."""
    cand = [c for c in all_cols if c not in (current_col, pred_col)]
    if not cand:
        raise ValueError("No candidate model reply column found (expected 1 non-key column).")
    # Prefer columns with reply-ish words
    priority = [c for c in cand if re.search(r"(reply|response|output|answer|prediction)", _norm(c))]
    return priority[0] if priority else cand[-1]  # last non-key as fallback

def _load_one(path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load one CSV, detect keys + reply col, standardize key names, and rename reply col to {model}_reply.
    Returns: (df_with_standard_cols, model_name)
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File {path} has no rows.")

    current_col = _find_key_col(df.columns, KEY_CANDIDATES_CURRENT)
    pred_col = _find_key_col(df.columns, KEY_CANDIDATES_PRED)
    reply_col = _choose_reply_col(list(df.columns), current_col, pred_col)

    # Try to infer model name; if empty, fallback to file stem
    model_name = _infer_model_from_reply_col(reply_col)
    if not model_name or model_name == reply_col:
        model_name = os.path.splitext(os.path.basename(path))[0]

    # Standardize keys to fixed names
    df = df.rename(columns={current_col: "current_action", pred_col: "predicted_action"})

    # Only keep the necessary columns
    df = df[["current_action", "predicted_action", reply_col]].copy()

    # Rename reply col to {model}_reply (sanitize model name to be column-safe)
    safe_model = re.sub(r"[^A-Za-z0-9_.\-]+", "_", model_name).strip("_")
    reply_out_col = f"{safe_model}_reply"
    df = df.rename(columns={reply_col: reply_out_col})

    # Deduplicate by keys (keep first)
    df = df.drop_duplicates(subset=["current_action", "predicted_action"], keep="first")

    return df, safe_model


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sample aligned rows across 3 model CSVs into one file.")
    ap.add_argument("csv1", help="Path to first model CSV")
    ap.add_argument("csv2", help="Path to second model CSV")
    ap.add_argument("csv3", help="Path to third model CSV")
    ap.add_argument("number_of_samples", type=int, help="Number of rows to sample")
    ap.add_argument("--output", "-o", default="sampled.csv", help="Output CSV path (default: sampled.csv)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    args = ap.parse_args()

    # Load files
    df1, m1 = _load_one(args.csv1)
    df2, m2 = _load_one(args.csv2)
    df3, m3 = _load_one(args.csv3)

    # Ensure model names unique
    m1, m2, m3 = _ensure_unique([m1, m2, m3])

    # After uniqueness fix, make sure columns match the (possibly updated) names
    # If uniqueness changed a name, rename the reply columns accordingly.
    for df, old_model, new_model in ((df1, df1.columns[2][:-6], m1), (df2, df2.columns[2][:-6], m2), (df3, df3.columns[2][:-6], m3)):
        old_reply_col = f"{old_model}_reply" if df.columns[2].endswith("_reply") else df.columns[2]
        if new_model and old_reply_col != f"{new_model}_reply":
            df.rename(columns={old_reply_col: f"{new_model}_reply"}, inplace=True)

    # Inner-join on keys so we only keep rows present in ALL three
    merged = df1.merge(df2, on=["current_action", "predicted_action"], how="inner")
    merged = merged.merge(df3, on=["current_action", "predicted_action"], how="inner")

    # Sample
    total = len(merged)
    if total == 0:
        raise SystemExit("No overlapping rows across the three files (by current_action & predicted_action). Nothing to sample.")
    n = min(args.number_of_samples, total)
    sampled = merged.sample(n=n, random_state=args.seed).reset_index(drop=True)

    # Add empty score columns for each model, and order columns as requested
    reply_cols = [c for c in sampled.columns if c.endswith("_reply")]
    # Order reply columns in the same order as inputs: m1, m2, m3
    ordered_reply_cols = [f"{m1}_reply", f"{m2}_reply", f"{m3}_reply"]

    for model in (m1, m2, m3):
        score_col = f"{model}_score"
        sampled[score_col] = ""  # empty for future filling

    ordered_cols = (
        ["current_action", "predicted_action"] +
        [col for pair in zip(ordered_reply_cols, [f"{m1}_score", f"{m2}_score", f"{m3}_score"]) for col in pair]
    )

    # In rare cases, if a reply column is missing due to unexpected input, fall back to existing order
    for col in ordered_cols:
        if col not in sampled.columns:
            # rebuild columns safely: just move keys first, then any *_reply, then *_score
            keys = ["current_action", "predicted_action"]
            replies = [c for c in sampled.columns if c.endswith("_reply")]
            scores = [c for c in sampled.columns if c.endswith("_score")]
            ordered_cols = keys + sum(([r, r.replace("_reply", "_score")] for r in replies), [])
            break

    sampled = sampled[ordered_cols]

    # Write output
    sampled.to_csv(args.output, index=False)
    print(f"Wrote {n} sampled rows to: {args.output} (from {total} overlapping rows).")


if __name__ == "__main__":
    main()
