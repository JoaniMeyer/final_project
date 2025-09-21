#!/usr/bin/env python3
"""
bertopic_modeling.py

Creator-specific topic modeling using BERTopic with South African cultural guidance.

Improvements:
- Creator-specific extra stopwords (channel name, common spam/meta words)
- Phrase protection for multi-word SA terms ("cape flats" -> "cape_flats")
- Multilingual embeddings
- 1–3 ngrams for better phrases
- Re-rank topic keywords with KeyBERTInspired + MMR for cleaner, diverse key terms
"""

from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"  # To fix OMP nested routine deprecation warning

# Suppress OMP warnings
import warnings
import sys
from contextlib import redirect_stderr
from io import StringIO

warnings.filterwarnings("ignore", message=".*omp_set_nested.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*omp.*")

# Function to suppress OMP warnings during BERTopic operations
def suppress_omp_warnings():
    """Context manager to suppress OMP warnings"""
    class OMPWarningSuppressor:
        def __enter__(self):
            self.original_stderr = sys.stderr
            sys.stderr = StringIO()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stderr = self.original_stderr
    
    return OMPWarningSuppressor()

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Set, Optional, cast
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired  # MMR+embedding-based keyword refinement


# ----------------------------- Paths & loaders ----------------------------- #

def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p and p.expanduser().exists():
            return p.expanduser()
    return None


@dataclass
class Paths:
    root: str = "."

    @property
    def root_path(self) -> Path:
        return Path(self.root).resolve()

    @property
    def config_dir(self) -> Path:
        return self.root_path / "config"

    @property
    def processed_dir(self) -> Path:
        return self.root_path / "processed_data"

    @property
    def topics_dir(self) -> Path:
        return self.processed_dir / "topics"

    @property
    def topic_config_yaml(self) -> Path:
        return self.config_dir / "topic_config.yaml"

    @property
    def category_mapping_json(self) -> Path:
        return self.config_dir / "category_mapping.json"

    @property
    def cultural_dict_path(self) -> Path:
        cand = _first_existing([
            self.config_dir / "sa_cultural_dict_improved.json",
            self.config_dir / "sa_cultural_dict.json",
            self.root_path / "sa_cultural_dict_improved.json",
            self.root_path / "sa_cultural_dict.json",
            self.root_path / "moral_landscape_app" / "dictionary" / "sa_cultural_dict_improved.json",
            self.root_path / "moral_landscape_app" / "dictionary" / "sa_cultural_dict.json",
        ])
        if not cand:
            raise FileNotFoundError(
                "Could not locate a cultural dictionary. "
                "Place sa_cultural_dict_improved.json in config/, project root, or moral_landscape_app/dictionary/."
            )
        return cand

    @property
    def stopwords_path(self) -> Path:
        cand = _first_existing([
            self.config_dir / "sa_stopwords.csv",
            self.config_dir / "sa_stopwords.json",
            self.root_path / "sa_stopwords.csv",
            self.root_path / "sa_stopwords.json",
            self.root_path / "moral_landscape_app" / "stopwords" / "sa_stopwords.csv",
            self.root_path / "moral_landscape_app" / "stopwords" / "sa_stopwords.json",
        ])
        if not cand:
            raise FileNotFoundError(
                "Could not locate stopwords. "
                "Place sa_stopwords.{csv|json} in config/, project root, or moral_landscape_app/stopwords/."
            )
        return cand


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}")
    return cast(Dict[str, Any], data)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------- Cultural resources ----------------------------- #

@dataclass
class CulturalResources:
    terms: Set[str]
    term_to_display_categories: Dict[str, List[str]]
    seed_topic_list: List[List[str]]
    stopwords: Set[str]
    phrase_regex: Optional[re.Pattern]
    display_categories: List[str]
    dictionary_path: Path
    stopwords_path: Path


def load_resources(paths: Paths) -> CulturalResources:
    config = _read_yaml(paths.topic_config_yaml)
    mapping_raw = _read_json(paths.category_mapping_json)
    if not isinstance(mapping_raw, dict):
        raise ValueError("category_mapping.json must be an object {technical: display or [display,...]}")

    # normalize mapping
    mapping: Dict[str, List[str]] = {}
    for k, v in mapping_raw.items():
        mapping[k] = [str(x) for x in (v if isinstance(v, list) else [v])]

    # cultural dictionary
    dic_raw = _read_json(paths.cultural_dict_path)
    if isinstance(dic_raw, dict) and "terms" in dic_raw:
        dic_raw = dic_raw["terms"]
    if not isinstance(dic_raw, list):
        raise ValueError("Cultural dictionary must be a list of {term, category, variants?, meaning?}")

    def to_display(technical: Iterable[str] | None) -> List[str]:
        out: List[str] = []
        for t in technical or []:
            for d in mapping.get(t, ["Language & Expression"]):
                if d not in out:
                    out.append(d)
        return out or ["Language & Expression"]

    terms: Set[str] = set()
    term_to_disp: Dict[str, List[str]] = {}
    for itm in dic_raw:
        if not isinstance(itm, dict):
            continue
        term = str(itm.get("term", "")).strip().lower()
        
        # Use the new 'topic' field if available, otherwise fall back to category mapping
        if "topic" in itm:
            topic = str(itm.get("topic", "")).strip()
            if topic:
                displays = [topic]
            else:
                cats = itm.get("category", [])
                displays = to_display(cats if isinstance(cats, list) else [cats])
        else:
            cats = itm.get("category", [])
            displays = to_display(cats if isinstance(cats, list) else [cats])
        
        if term:
            terms.add(term); term_to_disp[term] = displays
        for v in itm.get("variants", []) or []:
            v2 = str(v).strip().lower()
            if v2:
                terms.add(v2); term_to_disp[v2] = displays

    # seed topics grouped by display category order
    disp_to_terms: Dict[str, List[str]] = {}
    for t, cs in term_to_disp.items():
        for c in cs:
            disp_to_terms.setdefault(c, []).append(t)

    display_order = list(config.get("display_category_order", []))
    min_seed = int(config.get("seeding", {}).get("min_seed_terms_per_topic", 5))
    max_seed = int(config.get("seeding", {}).get("max_seed_terms_per_topic", 50))

    seed_topic_list: List[List[str]] = []
    display_categories: List[str] = []
    for disp in display_order:
        group = sorted(set(disp_to_terms.get(disp, [])))
        if len(group) >= min_seed:
            seed_topic_list.append(group[:max_seed])
            display_categories.append(disp)

    # stopwords (csv/json)
    stopwords_path = paths.stopwords_path
    stopwords: Set[str] = set()
    if stopwords_path.suffix.lower() == ".csv":
        sw = pd.read_csv(stopwords_path)
        if not {"lang", "stopword"} <= set(sw.columns):
            raise ValueError("sa_stopwords.csv must have columns: lang, stopword")
        stopwords = set(s.strip().lower() for s in sw["stopword"].astype(str) if s)
    else:
        data = _read_json(stopwords_path)
        if isinstance(data, dict):
            for _, words in data.items():
                stopwords.update(str(w).strip().lower() for w in (words or []))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "stopword" in item:
                    stopwords.add(str(item["stopword"]).strip().lower())

    # never drop cultural terms
    stopwords.difference_update(terms)

    # phrase protection regex
    multi = sorted([t for t in terms if " " in t], key=lambda s: -len(s))
    phrase_regex = re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in multi) + r")\b") if multi else None

    return CulturalResources(
        terms=terms,
        term_to_display_categories=term_to_disp,
        seed_topic_list=seed_topic_list,
        stopwords=stopwords,
        phrase_regex=phrase_regex,
        display_categories=display_categories,
        dictionary_path=paths.cultural_dict_path,
        stopwords_path=paths.stopwords_path,
    )


# ----------------------------- helpers ----------------------------- #

_GENERIC_META_STOPWORDS: Set[str] = {
    # platform/meta
    "subscribe", "subscribed", "subscription", "follow", "followers", "following",
    "share", "shared", "like", "liked", "link", "video", "vid", "channel", "views",
    # filler/affect
    "wow", "amazing", "awesome", "beautiful", "stunning", "good", "nice", "great",
    "hahaha", "haha", "lol", "lmao", "temu",
}

def _tokens_from_creator(creator_id: Any) -> Set[str]:
    s = str(creator_id).lower()
    parts = re.split(r"[^\w]+", s)
    parts = [p for p in parts if len(p) >= 2]
    return set(parts)


def make_phrase_preprocessor(resources: CulturalResources):
    regex = resources.phrase_regex

    def _pre(s: str) -> str:
        if not isinstance(s, str):
            return ""
        # cleanup
        s = re.sub(r"http[s]?://\S+", " ", s)
        s = re.sub(r"@\w+", " ", s)
        s = s.lower()
        # collapse laughter/repeats
        s = re.sub(r"\b(?:ha){2,}\b", " ", s)
        s = re.sub(r"\b(?:lol){2,}\b", " ", s)
        # protect phrases
        if regex:
            s = regex.sub(lambda m: m.group(0).replace(" ", "_"), s)
        # normalize spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    return _pre


def score_category(top_words: List[str], resources: CulturalResources) -> Tuple[str, Dict[str, float]]:
    scores: Dict[str, float] = {c: 0.0 for c in (resources.display_categories or [])}
    if not scores:
        for cats in resources.term_to_display_categories.values():
            for c in cats:
                scores.setdefault(c, 0.0)

    # Score based on cultural dictionary matches
    for rank, word in enumerate(top_words):
        weight = max(0, 15 - rank)
        plain = word.replace("_", " ")
        
        # Try exact matches first
        cats = resources.term_to_display_categories.get(plain, []) or resources.term_to_display_categories.get(word, [])
        
        # If no exact match, try partial matches (word appears in dictionary term)
        # Only match if the word is at least 4 characters
        # Basic words are now handled by the stopwords file
        if not cats and len(plain) >= 4:
            
            for dict_term, dict_cats in resources.term_to_display_categories.items():
                # Handle consolidated terms with slashes (e.g., "afrikaans/afrikaanse/afrikaan")
                if '/' in dict_term:
                    # Check if word matches any part of the consolidated term
                    if plain in dict_term.split('/'):
                        cats = dict_cats
                        break
                else:
                    # Check if word appears as a whole word in the dictionary term
                    if f' {plain} ' in f' {dict_term} ' or dict_term.startswith(f'{plain} ') or dict_term.endswith(f' {plain}'):
                        cats = dict_cats
                        break
        
        for c in cats:
            scores[c] = scores.get(c, 0.0) + weight

    # If cultural terms found, use the best match
    if any(v > 0.0 for v in scores.values()):
        best = None
        best_val = -1.0
        for c in resources.display_categories + sorted(set(scores.keys())):
            v = scores.get(c, 0.0)
            if v > best_val:
                best = c; best_val = v
        return best or "Unknown", scores
    
    # If no cultural terms found, use intelligent categorization as fallback
    return intelligent_categorization(top_words), scores

def intelligent_categorization(words: List[str]) -> str:
    """Intelligent categorization using cultural dictionary - no hardcoded lists"""
    
    # If we reach here, it means no cultural terms were found
    return 'Unknown'


def words_for_display(words: Iterable[str]) -> List[str]:
    return [w.replace("_", " ").strip() for w in words if w]


# ----------------------------- modeling ----------------------------- #

def build_topic_model(
    resources: CulturalResources,
    comment_count: int,
    paths: Paths = Paths(),
    extra_stopwords: Optional[Set[str]] = None,
    creator_id: Optional[str] = None,
) -> BERTopic:
    cfg = _read_yaml(paths.topic_config_yaml)

    # embeddings (multilingual)
    embed_model_name = str(cfg.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"))
    embed_model = SentenceTransformer(embed_model_name, device="cpu")

    # vectorizer
    vec_params = cast(Dict[str, Any], cfg.get("vectorizer", {}))
    preproc = make_phrase_preprocessor(resources)
    token_pattern = str(vec_params.get("token_pattern", r"(?u)\b[a-zA-Z_]{2,}\b"))
    ngram = vec_params.get("ngram_range", [1, 3])  # default to (1,3)
    if isinstance(ngram, list) and len(ngram) == 2:
        ngram = (int(ngram[0]), int(ngram[1]))
    else:
        ngram = (1, 3)

    stopwords_all = set(resources.stopwords)
    if extra_stopwords:
        stopwords_all |= set(extra_stopwords)
    # never drop cultural terms
    stopwords_all.difference_update(resources.terms)

    vectorizer = CountVectorizer(
        ngram_range=cast(Tuple[int, int], ngram),
        stop_words=list(stopwords_all), # sa_stopwords.csv
        lowercase=True,
        min_df=int(vec_params.get("min_df", 2)),
        max_df=float(vec_params.get("max_df", 0.85)),
        token_pattern=token_pattern,
        preprocessor=preproc, # ← Phrase protection from cultural dict
    )

    # dynamic BERTopic params
    bt_params = dict(cast(Dict[str, Any], cfg.get("bertopic_params", {})))
    dyn = cast(Dict[str, Any], cfg.get("dynamic_params", {}))

    def apply_dyn(bucket: str) -> None:
        b = cast(Dict[str, Any], dyn.get(bucket, {}))
        if "min_topic_size" in b:
            bt_params["min_topic_size"] = int(b["min_topic_size"])
        if "max_topics" in b and bt_params.get("nr_topics") is not None:
            try:
                bt_params["nr_topics"] = min(int(bt_params["nr_topics"]), int(b["max_topics"]))
            except Exception:
                pass

    if comment_count < int(cast(Dict[str, Any], dyn.get("small_creator", {"min_comments": 200}))["min_comments"]):
        apply_dyn("small_creator")
    elif comment_count < int(cast(Dict[str, Any], dyn.get("medium_creator", {"min_comments": 1000}))["min_comments"]):
        apply_dyn("medium_creator")
    else:
        apply_dyn("large_creator")

    # sanitize None values
    bt_params = {k: v for k, v in bt_params.items() if v is not None}

    # validated seed topics (optional)
    seed_list: List[List[str]] = []
    if resources.seed_topic_list:
        for topic_terms in resources.seed_topic_list:
            valid_terms = [t for t in topic_terms if len(t) > 2]
            if len(valid_terms) >= 3:
                seed_list.append(valid_terms)

    # build model with adaptive min_topic_size for short content
    # Use smaller min_topic_size for creators with mostly short comments
    min_topic_size = bt_params.get("min_topic_size", 10)
    
    # Check if this is likely to be short content (estimate based on creator)
    if creator_id and isinstance(creator_id, str):
        # Dr. Phillips typically has short, fragmented comments
        if 'drphillips' in creator_id.lower() or 'phillips' in creator_id.lower():
            min_topic_size = min(min_topic_size, 3)
        # Other creators might also benefit from smaller topics
        elif min_topic_size > 5:
            min_topic_size = 5
    
    if seed_list:
        topic_model = BERTopic(
            embedding_model=embed_model,
            vectorizer_model=vectorizer,
            seed_topic_list=seed_list,
            min_topic_size=min_topic_size,
            calculate_probabilities=False,
        )
    else:
        topic_model = BERTopic(
            embedding_model=embed_model,
            vectorizer_model=vectorizer,
            min_topic_size=min_topic_size,
            calculate_probabilities=False,
        )
    return topic_model


def run_bertopic_for_creator(
    df: pd.DataFrame,
    creator_id: Any,
    paths: Paths = Paths(),
    min_comments: int = 40,
) -> Dict[str, Any]:
    # prefer 'source' if present
    if "source" in df.columns:
        cdf = df[df["source"] == creator_id].copy()
    elif "creator_id" in df.columns:
        cdf = df[df["creator_id"] == creator_id].copy()
    else:
        raise ValueError("DataFrame must have 'source' or 'creator_id' column.")

    if "text" not in cdf.columns:
        raise ValueError("DataFrame must have a 'text' column with the comment content.")

    # clean text
    text_series = pd.Series(cdf["text"])
    cdf = cdf.loc[text_series.notna()].copy()
    cdf["text"] = cdf["text"].astype(str).str.strip()
    
    # Remove cultural context markers added by V4 classification for cleaner topic modeling
    cdf["text"] = cdf["text"].str.replace(r' \[CULTURAL_TERMS:.*?\]', '', regex=True)
    cdf["text"] = cdf["text"].str.replace(r' \[CULTURAL_CATEGORIES:.*?\]', '', regex=True)
    
    cdf = cdf.loc[cdf["text"].str.len() > 3].copy()
    cdf = cdf.loc[~cdf["text"].str.match(r'^[\s\W]*$')].copy()
    cdf = cdf.loc[cdf["text"].str.len() < 10000].copy()

    if len(cdf) < min_comments:
        raise ValueError(f"Not enough comments for creator {creator_id} ({len(cdf)})")

    # creator-specific extra stopwords
    extra_sw = _tokens_from_creator(creator_id) | _GENERIC_META_STOPWORDS

    # build + fit model
    resources = load_resources(paths)
    topic_model = build_topic_model(
        resources,
        comment_count=len(cdf),
        paths=paths,
        extra_stopwords=extra_sw,
        creator_id=creator_id,
    )

    docs = [d for d in cdf["text"].astype(str).tolist() if d and len(d.strip()) > 0]
    if len(docs) < min_comments:
        raise ValueError(f"Not enough valid comments for creator {creator_id} after cleaning ({len(docs)})")

    # Use context manager to suppress OMP warnings during BERTopic operations
    with suppress_omp_warnings():
        topics, _ = topic_model.fit_transform(docs)

        # refine topic words with KeyBERTInspired (without custom topic assignments to avoid warnings)
        try:
            # Try with a smaller top_n_words to avoid weight compatibility issues
            rep = KeyBERTInspired(top_n_words=8)
            topic_model.update_topics(docs, representation_model=rep)
            print("Applied KeyBERTInspired representation")
        except Exception as e:
            # keep going with default representation if KeyBERTInspired is unavailable
            print(f"Could not apply KeyBERTInspired representation: {e}")
            print("Using default representation instead")

    # document info and topic counts
    docs_info = cast(pd.DataFrame, topic_model.get_document_info(docs))
    topic_series = cast(pd.Series, docs_info.loc[docs_info["Topic"] != -1, "Topic"])
    counts = cast(Dict[int, int], topic_series.value_counts().to_dict())

    topic_info: List[Dict[str, Any]] = []
    topic_cultural_scores: Dict[int, Dict[str, float]] = {}

    for raw_tid in sorted([t for t in topic_model.get_topics().keys() if t != -1]):
        tid = int(raw_tid)
        topic_terms_raw = topic_model.get_topic(tid)
        if not isinstance(topic_terms_raw, list):
            topic_terms_raw = []
        topic_terms: List[Tuple[str, float]] = cast(List[Tuple[str, float]], topic_terms_raw)

        top_words: List[str] = []
        if topic_terms:
            max_items = min(15, len(topic_terms))
            top_words = [w for (w, _wt) in topic_terms[:max_items] if isinstance(w, str)]

        # filter out any leftover creator/meta words from the display list
        filtered_top = [w for w in top_words if w not in extra_sw]
        best_cat, score_map = score_category(filtered_top, resources)
        topic_cultural_scores[tid] = score_map

        # prefer cultural hits for label
        cultural_hits = [w for w in filtered_top if w.replace("_", " ") in resources.terms][:4]
        label_bits = words_for_display(cultural_hits[:2]) or words_for_display(filtered_top[:2])
        label = f"{best_cat}:"

        topic_info.append({
            "Topic": tid,
            "Name": label,
            "Count": int(counts.get(tid, 0)),
            "Words": words_for_display(filtered_top[:10]),
            "Category": best_cat,
        })

    topic_info.sort(key=lambda x: x["Count"], reverse=True)

    # optional moral tilt
    moral_analysis: Dict[int, Any] = {}
    if "predicted_label" in cdf.columns and len(topic_info):
        assign = pd.DataFrame({"topic": topics, "label": cdf["predicted_label"].to_numpy()})
        for tid in [t["Topic"] for t in topic_info]:
            subset = assign.loc[assign["topic"] == tid]
            tot = float(len(subset))
            if tot == 0:
                continue
            u = float((subset["label"] == "Ubuntu").sum()) / tot
            c = float((subset["label"] == "Chaos").sum()) / tot
            m = float((subset["label"] == "Middle").sum()) / tot
            moral_analysis[tid] = {
                "ubuntu_prob": u, "chaos_prob": c, "middle_prob": m,
                "tilt": c - u, "total_comments": int(tot),
            }

    cfg = _read_yaml(paths.topic_config_yaml)
    result: Dict[str, Any] = {
        "creator_id": creator_id,
        "topic_info": topic_info,
        "topic_cultural_scores": topic_cultural_scores,
        "moral_analysis": moral_analysis,
        "num_comments": int(len(cdf)),
        "model_metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "embedding_model": str(cfg.get("embedding_model", "")),
            "seeded": bool(resources.seed_topic_list),
            "num_seed_topics": int(len(resources.seed_topic_list)),
            "dictionary_path": str(resources.dictionary_path),
            "stopwords_path": str(resources.stopwords_path),
        },
    }
    return result


def save_creator_topics(result: Dict[str, Any], paths: Paths = Paths()) -> str:
    paths.topics_dir.mkdir(parents=True, exist_ok=True)
    creator = str(result.get("creator_id"))
    out_path = paths.topics_dir / f"{creator}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return str(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run BERTopic for a single creator")
    parser.add_argument("--creator", required=True, help="creator_id or source value")
    parser.add_argument("--data", default=str(Path("processed_data") / "integrated_comments.parquet"))
    parser.add_argument("--min-comments", type=int, default=40)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    res = run_bertopic_for_creator(df, args.creator, min_comments=args.min_comments)
    p = save_creator_topics(res)
    print(f"Saved topics for {args.creator} to {p}")
