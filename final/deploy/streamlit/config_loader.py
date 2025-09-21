#!/usr/bin/env python3
"""
config_loader.py
----------------
Single source of truth for configuration, dictionaries, stopwords and topic results.
This module keeps all paths and data contracts in one place so the UI and the
modeling code never disagree again.

What this module guarantees:
- Exactly one cultural dictionary is used (prefers *sa_cultural_dict_improved.json*).
- One category mapping is used to map technical labels -> display categories.
- Stopwords are loaded once and provided per language as sets.
- Creator topic results are read from disk and normalized into a stable schema.

Typical usage:

    from config_loader import (
        load_runtime_config, load_cultural_dictionary, load_category_mapping,
        load_stopwords, list_creators_with_topics, load_creator_topics,
        compute_cache_key
    )

All returned structures are plain Python dicts/lists for easy serialization.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import csv
import hashlib


# ---------- Paths & discovery ----------

BASE_DIR = Path(__file__).resolve().parent

# Default locations (searched in this order)
DICT_CANDIDATES = [
    BASE_DIR / "config" / "sa_cultural_dict_improved.json",
    BASE_DIR / "sa_cultural_dict_improved.json",
    BASE_DIR / "config" / "sa_cultural_dict.json",
    BASE_DIR / "sa_cultural_dict.json",
    BASE_DIR / "moral_landscape_app" / "dictionary" / "sa_cultural_dict.json",
]

STOPWORD_CANDIDATES = [
    BASE_DIR / "config" / "sa_stopwords.csv",
    BASE_DIR / "sa_stopwords.csv",
    BASE_DIR / "config" / "sa_stopwords.json",
    BASE_DIR / "sa_stopwords.json",
    BASE_DIR / "moral_landscape_app" / "stopwords" / "sa_stopwords.csv",
]

CATEGORY_MAP_CANDIDATES = [
    BASE_DIR / "config" / "category_mapping.json",
    BASE_DIR / "category_mapping.json",
]

TOPIC_RESULTS_DIR_CANDIDATES = [
    BASE_DIR / "processed_data" / "topics",          # per-creator JSON files
    BASE_DIR / "processed_data"                       # legacy single JSON
]

LEGACY_COMBINED_RESULTS = [
    "enhanced_topic_results.json",                    # may be dict keyed by creator
    "enhanced_topic_results_per_creator.json",        # alternate name
]


@dataclass
class RuntimeConfig:
    dict_path: Path
    stopwords_path: Optional[Path]
    category_mapping_path: Optional[Path]
    topics_dir: Optional[Path]
    legacy_results_file: Optional[Path]


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def load_runtime_config() -> RuntimeConfig:
    """
    Resolve and lock in the concrete files/folders we will use for:
    - cultural dictionary
    - stopwords
    - category mapping
    - topics location
    This function does not read the files; it only decides *which* ones to use.
    """
    dict_path = _first_existing(DICT_CANDIDATES)
    if not dict_path:
        raise FileNotFoundError(
            "Cultural dictionary not found. Expected one of:\n" +
            "\n".join(str(p) for p in DICT_CANDIDATES)
        )

    stopwords_path = _first_existing(STOPWORD_CANDIDATES)

    category_mapping_path = _first_existing(CATEGORY_MAP_CANDIDATES)

    topics_dir = _first_existing(TOPIC_RESULTS_DIR_CANDIDATES)

    legacy_file: Optional[Path] = None
    if topics_dir and topics_dir.name != "processed_data":
        # topics dir found; still check legacy in processed_data as a fallback
        legacy_root = BASE_DIR / "processed_data"
    else:
        legacy_root = topics_dir or (BASE_DIR / "processed_data")
    for fname in LEGACY_COMBINED_RESULTS:
        candidate = legacy_root / fname
        if candidate.exists():
            legacy_file = candidate
            break

    return RuntimeConfig(
        dict_path=dict_path,
        stopwords_path=stopwords_path,
        category_mapping_path=category_mapping_path,
        topics_dir=(BASE_DIR / "processed_data" / "topics") if (BASE_DIR / "processed_data" / "topics").exists() else None,
        legacy_results_file=legacy_file,
    )


# ---------- Cultural dictionary ----------

def load_cultural_dictionary(dict_path: Optional[Path] = None) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Load the South African cultural dictionary and normalize to:
        terms: Dict[str, List[str]]  -> term -> list of *display* categories
        meanings: Dict[str, str]     -> term -> meaning/description
    Accepts either the improved or legacy schema:
      - Improved: list of {term, category:[...], meaning, variants:[...]}
      - Legacy:   dict term -> {category(s), meaning, variants}
    Both are flattened with lowercased terms and variants.
    """
    cfg = load_runtime_config() if dict_path is None else None
    dict_file = dict_path or cfg.dict_path  # type: ignore

    with open(dict_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms: Dict[str, List[str]] = {}
    meanings: Dict[str, str] = {}

    def add_term(t: str, cats: List[str], meaning: str):
        t = t.strip().lower()
        if not t:
            return
        # allow multiple categories but keep unique order
        seen = set()
        deduped = [c for c in cats if not (c in seen or seen.add(c))]
        if deduped:
            terms[t] = deduped
        if meaning:
            meanings[t] = meaning

    if isinstance(data, list):
        # Improved list schema
        for item in data:
            term = (item.get("term") or "").lower()
            cats = item.get("category") or item.get("categories") or []
            if isinstance(cats, str):
                cats = [cats]
            meaning = item.get("meaning") or ""
            add_term(term, cats, meaning)
            # variants
            for v in item.get("variants", []):
                add_term(str(v).lower(), cats, meaning)
    elif isinstance(data, dict):
        # Legacy dict schema
        for term, meta in data.items():
            cats = meta.get("category") or meta.get("categories") or []
            if isinstance(cats, str):
                cats = [cats]
            meaning = meta.get("meaning") or ""
            add_term(str(term).lower(), cats, meaning)
            for v in meta.get("variants", []):
                add_term(str(v).lower(), cats, meaning)
    else:
        raise ValueError("Unsupported cultural dictionary format")

    return terms, meanings


# ---------- Category mapping ----------

DEFAULT_CATEGORY_MAPPING: Dict[str, str] = {
    # Map *technical* categories produced by modeling to *display* buckets
    "discourse_marker": "Language & Expression",
    "negative_expression": "Language & Expression",
    "emotional_expression": "Language & Expression",
    "conversation_marker": "Language & Expression",
    "greeting": "Social & Greeting",
    "social_interaction": "Social & Greeting",
    "person_reference": "Social & Greeting",
    "family_reference": "Social & Greeting",
    "age_reference": "Social & Greeting",
    "food_reference": "Food & Culture",
    "drink_reference": "Food & Culture",
    "cooking_reference": "Food & Culture",
    "music_genre": "Music & Entertainment",
    "dance_reference": "Music & Entertainment",
    "entertainment": "Music & Entertainment",
    "place_reference": "Place & Location",
    "geography": "Place & Location",
    "location": "Place & Location",
    "place_geography": "Place & Location",
    "sports_reference": "Sports & Recreation",
    "recreation": "Sports & Recreation",
    "natural_environment": "Sports & Recreation",
    "representation": "Sports & Recreation",
    "cultural_identity": "Identity & Philosophy",
    "philosophy": "Identity & Philosophy",
    "heritage": "Identity & Philosophy",
    "health_condition": "Health & Traditional",
    "traditional_medicine": "Health & Traditional",
    "wellness": "Health & Traditional",
    "work_reference": "Work & Economy",
    "money_reference": "Work & Economy",
    "economy": "Work & Economy",
    "confrontation": "Conflict & Confrontation",
    "aggression": "Conflict & Confrontation",
    "violence": "Conflict & Confrontation",
    "south_african": "General South African",
    "national_identity": "General South African",
}

def load_category_mapping(path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load mapping from technical categories -> display categories.
    If a custom config file exists, it is merged over the defaults.
    """
    mapping = DEFAULT_CATEGORY_MAPPING.copy()
    cfg = load_runtime_config() if path is None else None
    map_path = path or (cfg.category_mapping_path if cfg else None)

    if map_path and Path(map_path).exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                user_map = json.load(f)
            if not isinstance(user_map, dict):
                raise ValueError("category_mapping.json must be a JSON object")
            mapping.update(user_map)
        except Exception as e:
            # Non-fatal: fall back to defaults
            print(f"[config_loader] Warning: failed to read category mapping: {e}")

    return mapping


# ---------- Stopwords ----------

def _load_stopwords_csv(p: Path) -> Dict[str, set]:
    out: Dict[str, set] = {}
    with open(p, newline="", encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        if not {"lang", "stopword"} <= set(reader.fieldnames or []):
            raise ValueError("stopwords CSV must have 'lang' and 'stopword' columns")
        for row in reader:
            lang = (row["lang"] or "en").strip().lower()
            word = (row["stopword"] or "").strip().lower()
            if not word:
                continue
            out.setdefault(lang, set()).add(word)
    return out

def _load_stopwords_json(p: Path) -> Dict[str, set]:
    out: Dict[str, set] = {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for lang, words in data.items():
            out[str(lang).lower()] = {str(w).lower() for w in (words or [])}
    elif isinstance(data, list):
        # list of {lang, stopword}
        for item in data:
            lang = str(item.get("lang", "en")).lower()
            word = str(item.get("stopword", "")).lower()
            if word:
                out.setdefault(lang, set()).add(word)
    else:
        raise ValueError("Unsupported stopwords JSON format")
    return out

DEFAULT_STOPWORDS = {
    "en": {
        "the","a","an","and","or","but","in","on","at","to","for","of","with","by",
        "is","are","was","were","be","been","being","have","has","had","do","does","did",
        "will","would","could","should","may","might","must","can","this","that","these","those",
        "i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","her","its","our","their"
    },
    "af": {"die","n","en","van","is","op","vir","met","aan","te","om","dat","wat","nie","jy","ek","hy","sy","ons","hulle"},
}

def load_stopwords(path: Optional[Path] = None) -> Dict[str, set]:
    """
    Load multilingual stopwords. Returns dict[lang] -> set(words).
    Prefers CSV if both CSV and JSON exist. Falls back to a minimal default.
    """
    cfg = load_runtime_config() if path is None else None
    stop_path = path or (cfg.stopwords_path if cfg else None)

    if stop_path and stop_path.exists():
        try:
            if stop_path.suffix.lower() == ".csv":
                return _load_stopwords_csv(stop_path)
            else:
                return _load_stopwords_json(stop_path)
        except Exception as e:
            print(f"[config_loader] Warning: failed to read stopwords: {e}")

    # Fallback
    return {k: set(v) for k, v in DEFAULT_STOPWORDS.items()}


# ---------- Topic results ----------

def list_creators_with_topics(topics_dir: Optional[Path] = None) -> List[str]:
    """
    Enumerate creators for which per-creator topic files exist.
    Looks for processed_data/topics/<creator>.json
    """
    cfg = load_runtime_config() if topics_dir is None else None
    topics_root = topics_dir or (cfg.topics_dir if cfg else None)
    creators: List[str] = []
    if topics_root and topics_root.exists():
        for p in sorted(topics_root.glob("*.json")):
            creators.append(p.stem)
    else:
        # Fallback: inspect legacy combined file for keys
        if cfg and cfg.legacy_results_file and cfg.legacy_results_file.exists():
            try:
                data = json.loads(cfg.legacy_results_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    creators = list(data.keys())
            except Exception:
                pass
    return creators


def _normalize_topic_record(raw: Dict[str, Any], creator_id: str, default_category: str = "General") -> Dict[str, Any]:
    """
    Normalize a single raw topic record to our standard schema.
    Accepts multiple legacy key names (Words vs Top_Words, Name vs label, etc.)
    """
    words = raw.get("Words") or raw.get("Top_Words") or raw.get("keywords") or []
    if isinstance(words, str):
        words = [w.strip() for w in words.split(",")]
    name = raw.get("Name") or raw.get("label") or raw.get("TopicName") or ""
    category = raw.get("Category") or raw.get("category") or default_category
    return {
        "topic_id": str(raw.get("Topic", raw.get("id", ""))),
        "name": str(name),
        "count": int(raw.get("Count", raw.get("count", 0)) or 0),
        "words": [str(w) for w in words][:15],
        "category": str(category),
        "creator_id": creator_id,
    }


def load_creator_topics(creator_id: str, topics_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load topics for a single creator and normalize into:
    {
        "creator_id": ...,
        "total_comments": int,
        "topics": [ {topic_id, name, count, words, category, creator_id}, ... ],
        "meta": {...}
    }

    Supports:
    - per-creator file at processed_data/topics/<creator_id>.json
    - legacy combined file with a top-level dict keyed by creator_id
    - legacy BERTopic export formats
    """
    cfg = load_runtime_config() if topics_dir is None else None
    topics_root = topics_dir or (cfg.topics_dir if cfg else None)

    # Path #1: per-creator file
    if topics_root:
        f = topics_root / f"{creator_id}.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            return _normalize_creator_payload(creator_id, data)

    # Path #2: legacy combined results
    if cfg and cfg.legacy_results_file and cfg.legacy_results_file.exists():
        try:
            combined = json.loads(cfg.legacy_results_file.read_text(encoding="utf-8"))
            if isinstance(combined, dict) and creator_id in combined:
                return _normalize_creator_payload(creator_id, combined[creator_id])
        except Exception:
            pass

    raise FileNotFoundError(f"No topic results found for creator '{creator_id}'.")

def _normalize_creator_payload(creator_id: str, payload: Any) -> Dict[str, Any]:
    # A bunch of shapes we support...
    topics: List[Dict[str, Any]] = []
    total_comments = 0
    meta = {}

    if isinstance(payload, dict):
        # Possible shapes:
        # { 'topics': [...], 'total_comments': N, ... }
        # { 'topic_info': [...], 'num_comments': N, ... }
        # or even the whole structure under 'creator_results' etc.
        if "topics" in payload and isinstance(payload["topics"], list):
            topics = payload["topics"]
        elif "topic_info" in payload and isinstance(payload["topic_info"], list):
            topics = payload["topic_info"]
        elif "creator_results" in payload and creator_id in payload["creator_results"]:
            inner = payload["creator_results"][creator_id]
            topics = inner.get("topics") or inner.get("topic_info") or []

        total_comments = int(
            payload.get("total_comments")
            or payload.get("num_comments")
            or payload.get("creator_total_comments", 0)
        )

        # meta
        meta = {
            "timestamp": payload.get("timestamp") or payload.get("model_metadata", {}).get("timestamp"),
            "model_type": payload.get("model_type") or payload.get("model_metadata", {}).get("model_type"),
        }

    elif isinstance(payload, list):
        topics = payload
    else:
        raise ValueError("Unsupported creator payload format")

    norm = [_normalize_topic_record(t, creator_id) for t in topics]
    return {
        "creator_id": creator_id,
        "total_comments": total_comments,
        "topics": norm,
        "meta": meta,
    }


# ---------- Caching helpers ----------

def compute_cache_key(paths: List[Path]) -> str:
    """
    Robust cache key from file mtimes + sizes, so Streamlit can invalidate properly.
    """
    h = hashlib.sha256()
    for p in paths:
        try:
            stat = p.stat()
            h.update(str(p).encode("utf-8"))
            h.update(str(stat.st_mtime_ns).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
        except FileNotFoundError:
            h.update(f"missing:{p}".encode("utf-8"))
    return h.hexdigest()
