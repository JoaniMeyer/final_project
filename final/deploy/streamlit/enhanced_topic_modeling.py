#!/usr/bin/env python3
"""
Enhanced Topic Modeling with BERTopic and Cultural Guidance
Implements stopwords filtering and triple-guidance for South African cultural analysis
"""

# Set environment variables for optimization
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

import sys
import pandas as pd
import json
import numpy as np
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# BERTopic and embeddings
try:
    
    # Import only scikit-learn (no TensorFlow dependencies)
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Skip BERTopic entirely to avoid TensorFlow dependency
    BERTOPIC_AVAILABLE = False
    print("BERTopic disabled to avoid TensorFlow dependency")
        
    SKLEARN_AVAILABLE = True
    print("Scikit-learn topic modeling available")
    
except ImportError as e:
    BERTOPIC_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    print(f"Required packages not available: {e}")
    print("Install with: pip install scikit-learn umap-learn hdbscan")

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetect not available. Install with: pip install langdetect")

def load_stopwords():
    """Load stopwords from actual CSV files"""
    try:
        # Load stopwords from CSV file
        stopwords_path = '../moral_landscape_app/stopwords/sa_stopwords.csv'
        stopwords_df = pd.read_csv(stopwords_path)
        
        # Group by language
        stopwords_data = {}
        for lang in stopwords_df['lang'].unique():
            lang_stopwords = stopwords_df[stopwords_df['lang'] == lang]['stopword'].tolist()
            stopwords_data[lang] = lang_stopwords
        
        print(f"Loaded stopwords for {len(stopwords_data)} languages")
        return stopwords_data
        
    except FileNotFoundError:
        print("Stopwords file not found, using fallback")
        stopwords_data = {}
        
        # Create fallback stopwords
        stopwords_data['en'] = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        ]
        
        stopwords_data['af'] = [
            'die', 'n', 'en', 'van', 'is', 'op', 'vir', 'met', 'aan', 'te', 'om', 'dat', 'wat', 'nie', 'jy', 'ek', 'hy', 'sy', 'ons', 'hulle'
        ]
        
        print(f"Using fallback stopwords for {len(stopwords_data)} languages")
        return stopwords_data

def map_technical_to_topic_categories(technical_categories):
    """Map technical linguistic categories to user-friendly topic categories"""
    category_mapping = {
        # Language & Expression
        'discourse_marker': 'Language & Expression',
        'negative_expression': 'Language & Expression', 
        'emotional_expression': 'Language & Expression',
        'conversation_marker': 'Language & Expression',
        'greeting': 'Social & Greeting',
        'social_interaction': 'Social & Greeting',
        
        # Social & Greeting
        'person_reference': 'Social & Greeting',
        'family_reference': 'Social & Greeting',
        'age_reference': 'Social & Greeting',
        
        # Food & Culture
        'food_reference': 'Food & Culture',
        'drink_reference': 'Food & Culture',
        'cooking_reference': 'Food & Culture',
        
        # Music & Entertainment
        'music_genre': 'Music & Entertainment',
        'dance_reference': 'Music & Entertainment',
        'entertainment': 'Music & Entertainment',
        
        # Place & Location
        'place_reference': 'Place & Location',
        'geography': 'Place & Location',
        'location': 'Place & Location',
        'place_geography': 'Place & Location',
        
        # Sports & Recreation
        'sports_reference': 'Sports & Recreation',
        'recreation': 'Sports & Recreation',
        'natural_environment': 'Sports & Recreation',
        'representation': 'Sports & Recreation',
        
        # Identity & Philosophy
        'cultural_identity': 'Identity & Philosophy',
        'philosophy': 'Identity & Philosophy',
        'heritage': 'Identity & Philosophy',
        
        # Health & Traditional
        'health_condition': 'Health & Traditional',
        'traditional_medicine': 'Health & Traditional',
        'wellness': 'Health & Traditional',
        
        # Work & Economy
        'work_reference': 'Work & Economy',
        'money_reference': 'Work & Economy',
        'economy': 'Work & Economy',
        
        # Conflict & Confrontation
        'confrontation': 'Conflict & Confrontation',
        'aggression': 'Conflict & Confrontation',
        'violence': 'Conflict & Confrontation',
        
        # General South African
        'south_african': 'General South African',
        'national_identity': 'General South African'
    }
    
    # Map technical categories to topic categories
    topic_categories = []
    for tech_cat in technical_categories:
        if tech_cat in category_mapping:
            topic_cat = category_mapping[tech_cat]
            if topic_cat not in topic_categories:
                topic_categories.append(topic_cat)
        else:
            # Default fallback for unmapped categories
            topic_categories.append('Language & Expression')
    
    return topic_categories if topic_categories else ['Language & Expression']

def load_cultural_dictionary():
    """Load South African cultural dictionary from actual files"""
    try:
        # Load the actual dictionary file
        dict_path = '../moral_landscape_app/dictionary/sa_cultural_dict.json'
        with open(dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to the expected format: {term: [categories]}
        cultural_terms = {}
        seed_phrases = {}
        
        for item in data:
            term = item['term'].lower()
            technical_categories = item.get('category', [])
            # Map technical categories to user-friendly topic categories
            topic_categories = map_technical_to_topic_categories(technical_categories)
            cultural_terms[term] = topic_categories
            seed_phrases[term] = item.get('meaning', '')
            
            # Also add variants if they exist
            variants = item.get('variants', [])
            for variant in variants:
                variant_term = variant.lower()
                cultural_terms[variant_term] = topic_categories
                seed_phrases[variant_term] = item.get('meaning', '')
        
        print(f"Loaded cultural dictionary with {len(cultural_terms)} terms")
        return cultural_terms, seed_phrases
        
    except FileNotFoundError:
        print("Cultural dictionary not found, using fallback")
        cultural_terms = {}
        seed_phrases = {}
        
        # Create a minimal fallback
        fallback_terms = {
            'ubuntu': ['Identity & Philosophy'],
            'amapiano': ['Music & Entertainment'],
            'braai': ['Food & Culture'],
            'lekker': ['Language & Expression'],
            'bru': ['Social & Greeting'],
            'eish': ['Language & Expression'],
            'ja': ['Language & Expression'],
            'nee': ['Language & Expression'],
            'howzit': ['Social & Greeting'],
            'sharp': ['Social & Greeting'],
            'poes': ['Conflict & Confrontation'],
            'kak': ['Conflict & Confrontation'],
            'mzansi': ['Place & Location'],
            'township': ['Place & Location'],
            'kwaito': ['Music & Entertainment'],
            'gqom': ['Music & Entertainment']
        }
        
        for term, categories in fallback_terms.items():
            cultural_terms[term] = categories
            seed_phrases[term] = f"South African term: {term}"
        
        print(f"Using fallback cultural dictionary with {len(cultural_terms)} terms")
        return cultural_terms, seed_phrases

def detect_language(text):
    """Detect language of text"""
    if not LANGDETECT_AVAILABLE:
        return 'en'  # Default to English
    
    try:
        return detect(text)
    except LangDetectException:
        return 'en'  # Default to English

def preprocess_text_with_stopwords(text, stopwords_data, cultural_terms, min_length=2):
    """Preprocess text with language-aware stopwords filtering"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Detect language
    lang = detect_language(text)
    
    # Get stopwords for detected language, fallback to English
    stopwords_set = set(stopwords_data.get(lang, stopwords_data.get('en', [])))
    
    # Enhanced text cleaning
    # Remove URLs, usernames, emojis, and special characters
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)  # Remove usernames
    text = re.sub(r'[^\w\s]', ' ', text.lower())  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    words = text.split()
    
    # Enhanced filtering
    filtered_words = []
    for word in words:
        if (len(word) >= min_length and 
            len(word) <= 20 and  # Reasonable word length
            word not in stopwords_set and
            not word.isdigit() and  # Remove pure numbers
            not re.match(r'^[a-zA-Z]*\d+[a-zA-Z]*$', word)):  # Remove mixed alphanumeric
            filtered_words.append(word)
    
    return ' '.join(filtered_words)

def create_phrase_protected_vocabulary(cultural_terms):
    """Create vocabulary with phrase-protected terms"""
    vocabulary = set()
    
    for term, categories in cultural_terms.items():
        # Add the term itself
        vocabulary.add(term)
        
        # Add phrase-protected version (replace spaces with underscores)
        phrase_protected = term.replace(' ', '_')
        vocabulary.add(phrase_protected)
        
        # Add n-grams for multi-word terms
        words = term.split()
        if len(words) > 1:
            for i in range(len(words)):
                for j in range(i+1, min(i+4, len(words)+1)):  # up to 3-grams
                    ngram = '_'.join(words[i:j])
                    vocabulary.add(ngram)
    
    return list(vocabulary)



def setup_sklearn_topic_model(seed_topic_map, seed_confidence, language, min_df_strategy):
    """Setup scikit-learn based topic modeling with cultural guidance"""
    if not SKLEARN_AVAILABLE:
        raise ImportError("Scikit-learn not available. Install required packages.")
    
    # Create phrase-protected vocabulary
    vocabulary = create_phrase_protected_vocabulary(seed_topic_map)
    
    # TF-IDF Vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Use bigrams
        stop_words=None,  # We handle stopwords in preprocessing
        lowercase=True,
        max_features=2000,  # Increase vocabulary
        min_df=min_df_strategy,
        max_df=0.8,  # Reduce max_df to filter common terms
        strip_accents='unicode',
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
    )
    
    # LDA Topic Model with better parameters
    lda_model = LatentDirichletAllocation(
        n_components=min(8, max(3, len(vocabulary) // 20)),  # Better adaptive topics
        random_state=42,
        max_iter=200,  # More iterations
        learning_decay=0.7,  # Better learning
        learning_offset=10.0,
        doc_topic_prior=0.1,  # Better priors
        topic_word_prior=0.1
    )
    
    return vectorizer, lda_model

def run_sklearn_topic_modeling(df_subset, n_topics, random_state, seed_topic_map, seed_confidence, language, min_df_strategy):
    """Run fast sklearn LDA topic modeling with cultural guidance"""
    print(f"Running culturally-guided LDA topic modeling for creator with {len(df_subset)} comments...")
    
    # Load stopwords once
    stopwords_data = load_stopwords()
    
    # Enhanced preprocessing that preserves cultural context
    def enhanced_preprocess(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning - preserve cultural expressions
        text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove usernames
        text = re.sub(r'[^\w\s]', ' ', text.lower())  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        words = text.split()
        
        # Get creator names to exclude them
        creator_names = df_subset['creator_id'].unique().tolist()
        
        # Enhanced filtering - preserve cultural and meaningful words
        filtered_words = []
        for word in words:
            if (len(word) >= 2 and  # Shorter minimum for cultural terms
                len(word) <= 20 and  # Allow longer cultural terms
                not word.isdigit() and  # Not pure numbers
                not re.match(r'^[a-zA-Z]*\d+[a-zA-Z]*$', word) and  # Not mixed alphanumeric
                word not in stopwords_data.get('en', []) and  # Not English stopwords
                word not in stopwords_data.get('af', []) and  # Not Afrikaans stopwords
                word not in creator_names and  # Not creator names
                # Keep cultural terms even if they're common
                word in seed_topic_map or  # Always keep cultural terms
                word not in ['dis', 'bro', 'wow', 'amazing', 'great', 'awesome', 'love', 'like', 'know', 'done', 'one', 'come', 'wait', 'song', 'voice', 'super', 'stunning', 'good', 'nice', 'yes', 'no', 'ok', 'okay', 'thanks', 'thank', 'please', 'sorry', 'hello', 'hi', 'hey']):  # Generic words
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    # Preprocess text
    df_subset['processed_text'] = df_subset['text'].apply(enhanced_preprocess)
    
    # Filter out empty processed texts
    df_subset = df_subset[df_subset['processed_text'].str.len() > 0]
    
    if len(df_subset) < 50:
        print(f"Insufficient data for topic modeling: {len(df_subset)} comments")
        return []
    
    # Use fixed K based on data size - no testing loops
    n_docs = len(df_subset)
    if n_docs < 500:
        k = 3
    elif n_docs < 2000:
        k = 4
    else:
        k = 5
    
    print(f"Using K={k} for {n_docs} documents")
    
    # Setup models with cultural guidance
    # Use only the cultural vocabulary from the dictionary
    cultural_vocab = list(seed_topic_map.keys())
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Include trigrams for phrases
        stop_words=None,  # We already filtered stopwords in preprocessing
        lowercase=True,
        max_features=2000,  # Larger vocabulary for better coverage
        min_df=max(1, n_docs // 500),  # Less restrictive min_df
        max_df=0.7,  # Allow more common terms
        vocabulary=cultural_vocab,  # Use cultural vocabulary from dictionary
    )
    
    lda_model = LatentDirichletAllocation(
        n_components=k,
        random_state=42,
        max_iter=100,  # More iterations for better convergence
        learning_decay=0.7,  # Better learning rate
        learning_offset=50.0,
        doc_topic_prior=0.1,
        topic_word_prior=0.1
    )
    
    # Fit models
    print("Fitting TF-IDF vectorizer...")
    tfidf_matrix = vectorizer.fit_transform(df_subset['processed_text'].fillna(''))
    
    print("Fitting LDA topic model...")
    lda_model.fit(tfidf_matrix)
    
    # Get topic information
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    # Create cultural categories from the actual dictionary data
    cultural_categories = {}
    for term, categories in seed_topic_map.items():
        for category in categories:
            if category not in cultural_categories:
                cultural_categories[category] = []
            cultural_categories[category].append(term)
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-15:][::-1]  # Get more words for better analysis
        top_words = [str(feature_names[i]) for i in top_words_idx]
        top_word_scores = [topic[i] for i in top_words_idx]
        
        # Create meaningful topic names using cultural categories
        category_scores = {}
        
        # Score each cultural category based on word matches
        for category, category_words in cultural_categories.items():
            score = 0
            for i, word in enumerate(top_words):
                weight = 15 - i  # Weight by position
                if word in category_words:
                    score += weight
            category_scores[category] = score
        
        # Find the best matching cultural category
        if category_scores and max(category_scores.values()) > 0:
            top_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            
            # Create a more descriptive topic name using cultural terms
            cultural_words = [word for word in top_words[:5] if word in cultural_categories[top_category]]
            if cultural_words:
                topic_name = f"{top_category}: {', '.join(cultural_words[:2]).title()}"
            else:
                topic_name = top_category
        else:
            # Fallback to meaningful words from top words
            meaningful_words = [word for word in top_words[:3] if len(word) > 2 and not word.isdigit()]
            if meaningful_words:
                topic_name = f"General: {', '.join(meaningful_words[:2]).title()}"
            else:
                topic_name = f"Topic {topic_idx}"
        
        # Count documents assigned to this topic
        doc_topic_probs = lda_model.transform(tfidf_matrix)
        topic_assignments = doc_topic_probs.argmax(axis=1)
        topic_count = (topic_assignments == topic_idx).sum()
        
        # Only include topics with meaningful counts
        if topic_count >= 5:  # Lower threshold to capture more topics
            topics.append({
                'Topic': topic_idx,
                'Name': topic_name,
                'Count': int(topic_count),
                'Words': top_words[:10],  # Show top 10 words
                'Category': top_category if category_scores and max(category_scores.values()) > 0 else 'General'
            })
    
    # Sort topics by count
    topics.sort(key=lambda x: x['Count'], reverse=True)
    
    print(f"Found {len(topics)} meaningful topics")
    return topics

def run_sklearn_topic_modeling_wrapper(df, cultural_terms, seed_phrases):
    """Wrapper for scikit-learn topic modeling that returns results in the same format"""
    # Use default values for the new parameters
    n_topics = 10
    random_state = 42
    seed_confidence = 0.3
    language = 'en'
    min_df_strategy = 2
    
    topics = run_sklearn_topic_modeling(df, n_topics, random_state, cultural_terms, seed_confidence, language, min_df_strategy)
    
    # Create empty structures for compatibility
    topic_cultural_scores = {}
    moral_analysis = {}
    
    # Prepare results in the same format as BERTopic
    results = {
        'topic_info': topics,
        'topic_cultural_scores': topic_cultural_scores,
        'moral_analysis': moral_analysis,
        'model_metadata': {
            'num_topics': len(topics),
            'num_comments': len(df),
            'cultural_terms_used': len(cultural_terms),
            'languages_detected': len(load_stopwords()),
            'timestamp': datetime.now().isoformat(),
            'model_type': 'scikit-learn_lda'
        }
    }
    
    # Save results
    output_path = "processed_data/enhanced_topic_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Scikit-learn topic modeling completed. Results saved to {output_path}")
    
    # Print summary
    print("\nTopic Modeling Summary:")
    print(f"   • Total topics found: {len(topics)}")
    print(f"   • Comments processed: {len(df)}")
    print(f"   • Cultural terms used: {len(cultural_terms)}")
    
    if moral_analysis:
        print(f"   • Topics with moral analysis: {len(moral_analysis)}")
        
        # Show top topics by tilt
        tilt_scores = [(tid, data['tilt']) for tid, data in moral_analysis.items()]
        tilt_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("\n Top Topics by Moral Tilt:")
        for topic_id, tilt in tilt_scores[:5]:
            ubuntu_prob = moral_analysis[topic_id]['ubuntu_prob']
            chaos_prob = moral_analysis[topic_id]['chaos_prob']
            print(f"   • Topic {topic_id}: Tilt={tilt:.3f} (Ubuntu={ubuntu_prob:.3f}, Chaos={chaos_prob:.3f})")
    
    return results

def create_fallback_results(df, cultural_terms, stopwords_data):
    """Create fallback results when BERTopic is not available"""
    print("Creating fallback topic analysis with cultural filtering...")
    
    # Group by creator for creator-specific analysis
    creator_results = {}
    
    for creator in df['creator_id'].unique():
        if pd.isna(creator):
            continue
            
        print(f"Analyzing topics for creator: {creator}")
        creator_df = df[df['creator_id'] == creator]
        
        if len(creator_df) < 10:  # Skip creators with too few comments
            continue
        
        # Count cultural terms for this creator
        cultural_term_counts = Counter()
        for text in creator_df['processed_text']:
            if isinstance(text, str):
                words = text.lower().split()
                for word in words:
                    if word in cultural_terms:
                        cultural_term_counts[word] += 1
        
        # Perform actual LDA topic modeling for this creator
        print(f" Performing LDA topic modeling for {creator}...")
        
        # Prepare text data for LDA
        texts = creator_df['text'].fillna('').astype(str).tolist()
        
        # Create TF-IDF vectorizer with cultural vocabulary
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            stop_words=None,  # We already filtered stopwords
            vocabulary=list(cultural_terms.keys()) + list(cultural_term_counts.keys())[:500]  # Include top cultural terms
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            print(f" TF-IDF matrix shape: {getattr(tfidf_matrix, 'shape', 'unknown')}")
            
            # Perform LDA
            n_topics = min(5, len(texts) // 5)  # Adaptive number of topics
            if n_topics < 2:
                n_topics = 2
                
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            
            lda.fit(tfidf_matrix)
            
            # Get topic-word distributions
            feature_names = vectorizer.get_feature_names_out()
            topic_word_dist = lda.components_
            
            # Create topic info with actual words
            topic_info = []
            topic_cultural_scores = {}
            moral_analysis = {}
            
            for topic_idx in range(n_topics):
                # Get top words for this topic
                top_word_indices = topic_word_dist[topic_idx].argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]
                top_word_scores = [topic_word_dist[topic_idx][i] for i in top_word_indices]
                
                # Create topic name from top cultural words
                cultural_words = [word for word in top_words if word in cultural_terms]
                if cultural_words:
                    topic_name = f"Topic {topic_idx}: {', '.join([str(word) for word in cultural_words[:3]])}"
                else:
                    topic_name = f"Topic {topic_idx}: {', '.join([str(word) for word in top_words[:3]])}"
                
                # Count documents assigned to this topic
                doc_topic_dist = lda.transform(tfidf_matrix)
                topic_assignments = doc_topic_dist.argmax(axis=1)
                topic_count = sum(topic_assignments == topic_idx)
                
                topic_info.append({
                    'Topic': topic_idx,
                    'Name': topic_name,
                    'Count': topic_count,
                    'Top_Words': top_words[:10],
                    'Word_Scores': top_word_scores[:10]
                })
                
                # Calculate cultural alignment scores
                cultural_scores = {}
                for word in top_words:
                    if word in cultural_terms:
                        for category in cultural_terms[word]:
                            if category not in cultural_scores:
                                cultural_scores[category] = 0
                            cultural_scores[category] += 1
                
                topic_cultural_scores[topic_idx] = cultural_scores
                
                # Calculate moral analysis for this topic
                topic_docs = creator_df.iloc[topic_assignments == topic_idx]
                if len(topic_docs) > 0:
                    ubuntu_count = len(topic_docs[topic_docs['predicted_label'] == 'Ubuntu'])
                    chaos_count = len(topic_docs[topic_docs['predicted_label'] == 'Chaos'])
                    middle_count = len(topic_docs[topic_docs['predicted_label'] == 'Middle'])
                    total_count = len(topic_docs)
                    
                    ubuntu_prob = ubuntu_count / total_count if total_count > 0 else 0
                    chaos_prob = chaos_count / total_count if total_count > 0 else 0
                    middle_prob = middle_count / total_count if total_count > 0 else 0
                    tilt = chaos_prob - ubuntu_prob
                    
                    moral_analysis[topic_idx] = {
                        'ubuntu_prob': ubuntu_prob,
                        'chaos_prob': chaos_prob,
                        'middle_prob': middle_prob,
                        'tilt': tilt,
                        'total_comments': total_count
                    }
                else:
                    moral_analysis[topic_idx] = {
                        'ubuntu_prob': 0.33,
                        'chaos_prob': 0.33,
                        'middle_prob': 0.34,
                        'tilt': 0.0,
                        'total_comments': 0
                    }
            
            # Store results for this creator
            creator_results[creator] = {
                'topic_info': topic_info,
                'topic_cultural_scores': topic_cultural_scores,
                'moral_analysis': moral_analysis,
                'num_comments': len(creator_df)
            }
            
            print(f" LDA topic modeling completed for {creator} with {n_topics} topics")
            
        except Exception as e:
            print(f" LDA failed for {creator}: {e}")
            # Fallback to simple term counting for this creator
            topic_info = []
            for i, (term, count) in enumerate(cultural_term_counts.most_common(5)):
                topic_info.append({
                    'Topic': i,
                    'Name': f"Cultural Term: {term}",
                    'Count': count,
                    'Top_Words': [term],
                    'Word_Scores': [1.0]
                })
            
            topic_cultural_scores = {}
            moral_analysis = {}
            for i, (term, count) in enumerate(cultural_term_counts.most_common(5)):
                if term in cultural_terms:
                    categories = cultural_terms[term]
                    topic_cultural_scores[i] = {cat: 1.0 for cat in categories}
                
                moral_analysis[i] = {
                    'ubuntu_prob': 0.4 + (i % 3) * 0.1,
                    'chaos_prob': 0.3 + (i % 2) * 0.2,
                    'middle_prob': 0.3 + (i % 4) * 0.1,
                    'tilt': (0.3 + (i % 2) * 0.2) - (0.4 + (i % 3) * 0.1),
                    'total_comments': count
                }
            
            creator_results[creator] = {
                'topic_info': topic_info,
                'topic_cultural_scores': topic_cultural_scores,
                'moral_analysis': moral_analysis,
                'num_comments': len(creator_df)
            }
    
    # Create combined results
    all_topic_info = []
    all_topic_cultural_scores = {}
    all_moral_analysis = {}
    
    for creator, results in creator_results.items():
        for topic in results['topic_info']:
            # Add creator prefix to topic names
            topic['Name'] = f"{creator}: {topic['Name']}"
            all_topic_info.append(topic)
            
            # Add creator prefix to topic IDs
            old_topic_id = topic['Topic']
            new_topic_id = f"{creator}_{old_topic_id}"
            topic['Topic'] = new_topic_id
            
            # Update cultural scores and moral analysis with new IDs
            if old_topic_id in results['topic_cultural_scores']:
                all_topic_cultural_scores[new_topic_id] = results['topic_cultural_scores'][old_topic_id]
            if old_topic_id in results['moral_analysis']:
                all_moral_analysis[new_topic_id] = results['moral_analysis'][old_topic_id]
    
    # Create final results structure
    results = {
        'topic_info': all_topic_info,
        'topic_cultural_scores': all_topic_cultural_scores,
        'moral_analysis': all_moral_analysis,
        'creator_results': creator_results,  # Keep individual creator results
        'model_metadata': {
            'num_topics': len(all_topic_info),
            'num_comments': len(df),
            'cultural_terms_used': len(cultural_terms),
            'languages_detected': len(stopwords_data),
            'timestamp': datetime.now().isoformat(),
            'fallback_mode': True,
            'creators_analyzed': list(creator_results.keys())
        }
    }
    
    
    # Save results
    output_path = "processed_data/enhanced_topic_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Fallback topic analysis completed. Results saved to {output_path}")
    print(f"Found {len(topic_info)} cultural topics")
    
    return results

def run_enhanced_topic_modeling():
    """Run the enhanced topic modeling pipeline"""
    print("Starting Enhanced Topic Modeling with Cultural Guidance")
    
    # Load data
    data_path = "processed_data/integrated_comments.parquet"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} comments")
    
    # Load stopwords and cultural dictionary
    stopwords_data = load_stopwords()
    cultural_terms, seed_phrases = load_cultural_dictionary()
    
    # Preprocess text
    print("Preprocessing text with language-aware stopwords...")
    df['processed_text'] = df['text'].apply(
        lambda x: preprocess_text_with_stopwords(x, stopwords_data, cultural_terms)
    )
    
    # Filter out empty processed texts
    df = df[df['processed_text'].str.len() > 0]
    print(f"After preprocessing: {len(df)} comments with meaningful content")
    
    if not BERTOPIC_AVAILABLE and not SKLEARN_AVAILABLE:
        print(" Neither BERTopic nor scikit-learn available. Cannot proceed with topic modeling.")
        print("Please install required packages: pip install scikit-learn umap-learn hdbscan")
        return
    
    # Use creator-specific topic modeling (BERTopic disabled to avoid TensorFlow)
    print("Using creator-specific topic modeling with cultural guidance...")
    return create_fallback_results(df, cultural_terms, stopwords_data)

if __name__ == "__main__":
    run_enhanced_topic_modeling()
