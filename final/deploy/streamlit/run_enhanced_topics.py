#!/usr/bin/env python3
"""
run_enhanced_topics.py

Runner script for per-creator BERTopic modeling with cultural guidance.
Writes:
  - processed_data/topics/<creator>.json
  - processed_data/enhanced_topic_results.json
"""
from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"  # Fix OMP nested routine deprecation warning

# Suppress OMP warnings
import warnings
warnings.filterwarnings("ignore", message=".*omp_set_nested.*")

import json
import argparse
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path

import pandas as pd

from bertopic_modeling import (
    Paths,
    run_bertopic_for_creator,
    save_creator_topics,
    load_resources,
)


def _ensure_dirs(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.topics_dir.mkdir(parents=True, exist_ok=True)


def _load_df(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        try:
            df = pd.read_parquet(p)
        except Exception:
            df = pd.read_csv(p)

    # Normalize column names we need
    cols_lower = {c.lower(): c for c in df.columns}
    if "creator_id" not in df.columns and "source" not in df.columns:
        if "creator_id" in cols_lower:
            df.rename(columns={cols_lower["creator_id"]: "creator_id"}, inplace=True)
        elif "source" in cols_lower:
            df.rename(columns={cols_lower["source"]: "source"}, inplace=True)
        else:
            raise ValueError("Data must contain a 'creator_id' or 'source' column.")
    if "text" not in df.columns:
        if "text" in cols_lower:
            df.rename(columns={cols_lower["text"]: "text"}, inplace=True)
        else:
            raise ValueError("Data must contain a 'text' column (the comment content).")

    return df


def _list_creators(df: pd.DataFrame, min_comments: int) -> List[Any]:
    col = "source" if "source" in df.columns else "creator_id"
    counts = df[col].value_counts()
    return [idx for idx, cnt in counts.items() if int(cnt) >= int(min_comments)]


def _creator_file(paths: Paths, creator_id: Any) -> Path:
    return paths.topics_dir / f"{creator_id}.json"


def _should_skip(paths: Paths, creator_id: Any, data_path: str, force: bool) -> bool:
    if force:
        return False
    out = _creator_file(paths, creator_id)
    if not out.exists():
        return False
    try:
        return out.stat().st_mtime >= Path(data_path).stat().st_mtime
    except Exception:
        return True


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_all_creators(
    data_path: str = str(Path("processed_data") / "integrated_comments.parquet"),
    min_comments: int = 40,
    limit: int | None = None,
    force: bool = False,
    paths: Paths = Paths(),
) -> Dict[str, Any]:
    _ensure_dirs(paths)

    df = _load_df(data_path)
    creators = _list_creators(df, min_comments=min_comments)
    if limit is not None:
        creators = creators[: int(limit)]

    print(f"🚀 Running BERTopic for {len(creators)} creators (min_comments={min_comments})")
    print(f"📄 Data: {data_path}")
    print(f"📁 Output dir: {paths.topics_dir}")
    if force:
        print("⚠️  Force recompute enabled (ignoring cache)")

    # Pre-load resources once to fail fast if config/dicts missing
    _ = load_resources(paths)

    processed = 0
    errors: Dict[str, str] = {}
    for i, creator in enumerate(creators, start=1):
        try:
            if _should_skip(paths, creator, data_path, force=force):
                print(f"[{i}/{len(creators)}] ⏩ Skipping {creator} (cached)")
                continue

            print(f"[{i}/{len(creators)}] 🔄 Processing creator: {creator}")
            result = run_bertopic_for_creator(df, creator, paths=paths, min_comments=min_comments)
            out_path = save_creator_topics(result, paths=paths)
            print(f"   ✅ Saved: {out_path}")
            processed += 1
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"   ❌ Error on {creator}: {msg}")
            errors[str(creator)] = msg

    # build combined results by reading all per-creator files we have
    combined: Dict[str, Any] = {}
    for creator in creators:
        p = _creator_file(paths, creator)
        if p.exists():
            try:
                combined[str(creator)] = _read_json(p)
            except Exception as e:
                print(f"   ⚠️ Could not read {p}: {e}")

    combined_meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_path": data_path,
        "creators_total": len(creators),
        "creators_processed": processed,
        "errors": errors,
    }

    out_combined = paths.processed_dir / "enhanced_topic_results.json"
    with open(out_combined, "w", encoding="utf-8") as f:
        json.dump({"meta": combined_meta, "results": combined}, f, ensure_ascii=False, indent=2)
    print(f"\n📦 Combined results saved to {out_combined}")

    if errors:
        print("⚠️ Some creators failed:")
        for k, v in errors.items():
            print(f"   - {k}: {v}")

    return {"meta": combined_meta, "results": combined}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERTopic for all creators")
    parser.add_argument(
        "--data",
        default=str(Path("processed_data") / "integrated_comments.parquet"),
        help="Path to integrated comments parquet/csv",
    )
    parser.add_argument("--min-comments", type=int, default=40, help="Minimum comments per creator")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N creators")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    args = parser.parse_args()

    run_all_creators(
        data_path=args.data,
        min_comments=args.min_comments,
        limit=args.limit,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
