#!/usr/bin/env python3
"""
Advanced Integrated Topic Modeling Script
Combines LDA with cultural dictionary for sophisticated analysis
Shows per-video topics and creates interactive visualizations
"""

import sys
import os
import pandas as pd
import json
import numpy as np
import re
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def setup_nltk():
    """Setup NLTK data for multiple languages"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        return True
    except:
        return False

def get_multilingual_stopwords():
    """Get comprehensive stopwords for multiple languages including South African languages"""
    stopwords_set = set()
    
    # English stopwords
    try:
        nltk_stopwords = list(stopwords.words('english'))
        stopwords_set.update(nltk_stopwords)
    except:
        pass
    
    # Afrikaans stopwords (common words)
    afrikaans_stopwords = {
        'die', 'van', 'en', 'is', 'in', 'op', 'vir', 'met', 'aan', 'dat', 'wat', 'nie', 'om', 'te', 'sy', 'hy', 'ek', 'jy', 'ons', 'hulle', 'julle',
        'was', 'het', 'kan', 'sal', 'moet', 'wil', 'gaan', 'kom', 'sê', 'maar', 'ook', 'al', 'nog', 'net', 'so', 'baie', 'meer', 'selfs',
        'dan', 'as', 'of', 'tot', 'by', 'oor', 'uit', 'na', 'vir', 'deur', 'tussen', 'onder', 'bo', 'agter', 'voor', 'langs', 'binne',
        'buite', 'sonder', 'behalwe', 'tensy', 'indien', 'wanneer', 'waar', 'hoe', 'waarom', 'watter', 'wie', 'waarvan', 'waartoe'
    }
    stopwords_set.update(afrikaans_stopwords)
    
    # Zulu/Xhosa common words
    zulu_xhosa_stopwords = {
        'ukuthi', 'ukuba', 'ukuthi', 'ukuba', 'ukuthi', 'ukuba', 'ukuthi', 'ukuba', 'ukuthi', 'ukuba',
        'ngoba', 'kodwa', 'futhi', 'kakhulu', 'kakhulu', 'kakhulu', 'kakhulu', 'kakhulu', 'kakhulu',
        'lapha', 'lapho', 'lapha', 'lapho', 'lapha', 'lapho', 'lapha', 'lapho', 'lapha', 'lapho'
    }
    stopwords_set.update(zulu_xhosa_stopwords)
    
    # Social media and video platform noise
    social_media_noise = {
        'video', 'like', 'subscribe', 'comment', 'share', 'follow', 'watch', 'view', 'click',
        'link', 'channel', 'youtube', 'tiktok', 'instagram', 'facebook', 'twitter', 'social',
        'content', 'creator', 'influencer', 'viral', 'trending', 'fyp', 'foryou', 'reels',
        'story', 'post', 'upload', 'stream', 'live', 'broadcast', 'episode', 'series',
        'wow', 'amazing', 'beautiful', 'awesome', 'incredible', 'fantastic', 'great', 'good',
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'love', 'like',
        'yes', 'no', 'maybe', 'sure', 'ok', 'okay', 'alright', 'fine', 'nice', 'cool',
        'lol', 'haha', 'lmao', 'rofl', 'omg', 'wtf', 'fyi', 'btw', 'imo', 'tbh'
    }
    stopwords_set.update(social_media_noise)
    
    return stopwords_set

def load_cultural_dictionary():
    """Load and enhance the South African cultural dictionary"""
    try:
        with open('sa_dictionary.json', 'r', encoding='utf-8') as f:
            cultural_dict = json.load(f)
        
        sa_cultural_terms = set()
        if isinstance(cultural_dict, dict):
            for key, terms in cultural_dict.items():
                if isinstance(terms, list):
                    for term in terms:
                        if isinstance(term, str) and term.strip():
                            sa_cultural_terms.add(term.lower().strip())
        
        print(f"Loaded {len(sa_cultural_terms)} cultural terms")
        return sa_cultural_terms
    except Exception as e:
        print(f"Could not load cultural dictionary: {e}")
        return set()

def preprocess_text_advanced(text, cultural_terms, stopwords_set):
    """Advanced text preprocessing with cultural term preservation"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    words = text.split()
    
    # Enhanced filtering with cultural term preservation
    filtered_words = []
    for word in words:
        if (len(word) >= 3 and 
            word not in stopwords_set and
            not word.isdigit() and
            word.isalpha() and
            len(word) <= 20):
            filtered_words.append(word)
    
    # Add cultural context markers for important terms
    cultural_markers = []
    for term in cultural_terms:
        if term in text and len(term) > 3:
            cultural_markers.append(f"cultural_{term}")
    
    result = " ".join(filtered_words)
    if cultural_markers:
        result += " " + " ".join(cultural_markers)
    
    return result

def analyze_per_video_topics(df, source, cultural_terms, stopwords_set):
    """Analyze topics for each video within a source"""
    source_data = df[df['source'] == source].copy()
    
    # Group by video
    video_analysis = {}
    
    for video_id in source_data['videoWebUrl'].unique():
        video_data = source_data[source_data['videoWebUrl'] == video_id]
        video_comments = video_data['text'].dropna().tolist()
        
        if len(video_comments) < 3:  # Lower threshold to analyze more videos
            continue
        
        # Group comments by moral framework
        framework_comments = {'Chaos': [], 'Ubuntu': [], 'Middle': []}
        
        for i, comment in enumerate(video_comments):
            if i < len(video_data):
                moral_label = video_data.iloc[i]['moral_label']
                if moral_label in framework_comments:
                    framework_comments[moral_label].append(comment)
        
        # Analyze topics for each framework in this video
        video_topics = {}
        for framework_name, comments in framework_comments.items():
            if len(comments) >= 2:  # Lower threshold to get more framework analysis
                # Preprocess comments
                processed_comments = [preprocess_text_advanced(comment, cultural_terms, stopwords_set) 
                                    for comment in comments]
                processed_comments = [comment for comment in processed_comments if len(comment.split()) >= 3]
                
                if len(processed_comments) >= 2:  # Lower threshold for more analysis
                    # Use TF-IDF for better topic modeling with improved parameters
                    vectorizer = TfidfVectorizer(
                        max_features=500,
                        min_df=1,
                        max_df=0.7,
                        ngram_range=(1, 3),  # Include trigrams for better context
                        stop_words=list(stopwords_set),
                        lowercase=True,
                        strip_accents='unicode',
                        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
                    )
                    
                    try:
                        tfidf_matrix = vectorizer.fit_transform(processed_comments)
                        
                        # Use LDA for topic modeling with better parameters
                        n_topics = min(5, max(2, len(processed_comments) // 3))  # More topics for better granularity
                        lda = LatentDirichletAllocation(
                            n_components=n_topics,
                            random_state=42,
                            max_iter=200,  # More iterations for better convergence
                            learning_method='online',
                            learning_offset=50.0,
                            doc_topic_prior=0.05,  # Lower prior for more focused topics
                            topic_word_prior=0.05,  # Lower prior for more focused topics
                            verbose=0
                        )
                        
                        lda.fit(tfidf_matrix)
                        
                        # Extract topics
                        feature_names = vectorizer.get_feature_names_out()
                        topics = []
                        
                        for topic_idx, topic in enumerate(lda.components_):
                            top_words_idx = topic.argsort()[-10:][::-1]
                            top_words = [feature_names[i] for i in top_words_idx]
                            
                            # Filter meaningful words and preserve cultural terms
                            meaningful_words = []
                            cultural_words = []
                            for word in top_words:
                                word_str = str(word)
                                if word_str.startswith('cultural_'):
                                    # Extract cultural term
                                    cultural_term = word_str.replace('cultural_', '')
                                    cultural_words.append(f"{cultural_term}")  # Mark cultural terms
                                elif (len(word_str) > 3 and 
                                      word_str.isalpha() and
                                      word_str not in stopwords_set):
                                    # Check if this word is in our cultural dictionary
                                    if word_str in cultural_terms:
                                        meaningful_words.append(f"{word_str}")  # Mark cultural terms
                                    else:
                                        meaningful_words.append(word_str)
                            
                            # Combine meaningful words with cultural terms
                            if cultural_words:
                                meaningful_words = cultural_words + meaningful_words
                            
                            if len(meaningful_words) >= 3:
                                topics.append(meaningful_words[:5])
                        
                        video_topics[framework_name] = topics
                        
                    except Exception as e:
                        print(f"Error in LDA for {framework_name}: {e}")
                        # Fallback to simple word frequency
                        all_words = []
                        for comment in processed_comments:
                            all_words.extend(comment.split())
                        
                        word_counts = Counter(all_words)
                        common_words = [word for word, count in word_counts.most_common(10) 
                                      if len(word) > 3 and word not in stopwords_set and word.isalpha()]
                        
                        if common_words:
                            video_topics[framework_name] = [common_words[:5]]
        
        if video_topics:
            # Get timestamp with better handling
            timestamp = None
            if len(video_data) > 0:
                try:
                    timestamp = pd.to_datetime(video_data['timestamp'].iloc[0])
                except:
                    # If timestamp parsing fails, use a default timestamp
                    timestamp = pd.Timestamp.now()
            
            video_analysis[video_id] = {
                'topics': video_topics,
                'total_comments': len(video_comments),
                'timestamp': timestamp
            }
    
    return video_analysis

def get_video_details_for_selector(df, source):
    """Get video details for the video selector"""
    source_data = df[df['source'] == source].copy()
    
    video_details = []
    for video_url in source_data['videoWebUrl'].unique():
        video_data = source_data[source_data['videoWebUrl'] == video_url]
        
        # Get video info
        comment_count = len(video_data)
        timestamp = video_data['timestamp'].iloc[0] if len(video_data) > 0 else None
        
        # Get moral framework distribution
        moral_counts = video_data['moral_label'].value_counts().to_dict()
        
        video_details.append({
            'video_url': video_url,
            'comment_count': comment_count,
            'timestamp': timestamp,
            'moral_distribution': moral_counts,
            'display_name': f"Video {len(video_details) + 1} ({comment_count} comments)"
        })
    
    # Sort by timestamp
    video_details.sort(key=lambda x: x['timestamp'] if x['timestamp'] else pd.Timestamp.min)
    
    return video_details

def get_video_conversations(df, source, selected_video_url):
    """Get full conversation threads for a specific video"""
    video_data = df[(df['source'] == source) & (df['videoWebUrl'] == selected_video_url)].copy()
    
    # Sort by timestamp to show chronological order
    video_data = video_data.sort_values('timestamp')
    
    conversations = []
    for _, comment in video_data.iterrows():
        conversations.append({
            'text': comment['text'],
            'timestamp': comment['timestamp'],
            'moral_label': comment['moral_label'],
            'ubuntu_score': comment['proba_Ubuntu'],
            'chaos_score': comment['proba_Chaos'],
            'middle_score': comment['proba_Middle'],
            'digg_count': comment['diggCount'],
            'reply_count': comment['replyCommentTotal']
        })
    
    return conversations

def create_interactive_topic_evolution(video_analysis, source):
    """Create interactive chart showing topic evolution over time"""
    # Prepare data for visualization
    video_data = []
    
    for video_id, analysis in video_analysis.items():
        timestamp = analysis['timestamp']
        if timestamp:
            for framework, topics in analysis['topics'].items():
                for topic_idx, topic_words in enumerate(topics):
                    video_data.append({
                        'video_id': video_id,
                        'timestamp': timestamp,
                        'framework': framework,
                        'topic_id': f"{framework}_topic_{topic_idx}",
                        'topic_words': ', '.join(topic_words),
                        'comment_count': analysis['total_comments']
                    })
    
    if not video_data:
        return None
    
    # Create DataFrame
    df_viz = pd.DataFrame(video_data)
    df_viz['timestamp'] = pd.to_datetime(df_viz['timestamp'])
    df_viz = df_viz.sort_values('timestamp')
    
    # Add video index for better visualization
    df_viz['video_index'] = df_viz.groupby('timestamp').cumcount() + 1
    
    # Create interactive scatter plot with better time handling
    fig = px.scatter(
        df_viz, 
        x='timestamp', 
        y='framework',
        color='framework',
        size='comment_count',
        hover_data=['topic_words', 'comment_count', 'video_id'],
        title=f'Topic Evolution Over Time - {source}',
        color_discrete_map={
            'Chaos': '#d62728',
            'Ubuntu': '#2ca02c', 
            'Middle': '#ff7f0e'
        },
        labels={
            'timestamp': 'Time',
            'framework': 'Moral Framework',
            'comment_count': 'Comments'
        }
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def compute_advanced_topic_analysis():
    """Main function to compute advanced topic analysis"""
    print("🚀 Starting advanced integrated topic modeling...")
    
    # Setup
    nltk_available = setup_nltk()
    stopwords_set = get_multilingual_stopwords()
    cultural_terms = load_cultural_dictionary()
    
    # Load data
    try:
        df = pd.read_parquet('processed_data/integrated_comments.parquet')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} comments for advanced analysis")
    except Exception as e:
        print(f"Could not load data: {e}")
        return
    
    # Analyze each source
    results = {}
    
    for source in df['source'].unique():
        print(f"Advanced analysis for {source}...")
        
        # Per-video analysis
        video_analysis = analyze_per_video_topics(df, source, cultural_terms, stopwords_set)
        
        # Get video details for selector
        video_details = get_video_details_for_selector(df, source)
        
        # Create interactive visualization
        interactive_chart = create_interactive_topic_evolution(video_analysis, source)
        
        # Store results
        results[source] = {
            'video_analysis': video_analysis,
            'video_details': video_details,
            'total_videos': len(video_analysis),
            'interactive_chart': interactive_chart
        }
        
        print(f"  Analyzed {len(video_analysis)} videos")
    
    # Save results
    output_file = 'processed_data/advanced_topic_results.json'
    
    # Convert plotly figures to JSON for storage
    serializable_results = {}
    for source, data in results.items():
        serializable_results[source] = {
            'video_analysis': data['video_analysis'],
            'video_details': data['video_details'],
            'total_videos': data['total_videos']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Advanced topic analysis complete! Results saved to {output_file}")
    
    # Print summary
    for source, data in results.items():
        print(f"\n{source}:")
        print(f"   Total videos analyzed: {data['total_videos']}")
        
        # Show sample topics
        sample_videos = list(data['video_analysis'].keys())[:3]
        for video_id in sample_videos:
            video_data = data['video_analysis'][video_id]
            print(f"   Video {video_id[:8]}...:")
            for framework, topics in video_data['topics'].items():
                if topics:
                    print(f"     {framework}: {topics[0]}")

if __name__ == "__main__":
    compute_advanced_topic_analysis()
