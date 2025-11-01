"""
Feature engineering for tweet polarization prediction.
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


def extract_sentiment_features(df):
    """
    Extract sentiment-based features from tweets.
    
    Args:
        df: DataFrame with 'cleaned_text' column
        
    Returns:
        DataFrame with added sentiment features
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # VADER sentiment scores
    vader_scores = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x))
    df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    
    # TextBlob sentiment
    df['textblob_polarity'] = df['cleaned_text'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    df['textblob_subjectivity'] = df['cleaned_text'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )
    
    return df


def extract_engagement_features(df):
    """
    Extract engagement-based features.
    
    Args:
        df: DataFrame with engagement columns
        
    Returns:
        DataFrame with aggregated engagement features
    """
    features = {
        'total_retweets': df['retweet_count'].sum(),
        'total_likes': df['like_count'].sum(),
        'total_replies': df['reply_count'].sum(),
        'avg_retweets': df['retweet_count'].mean(),
        'avg_likes': df['like_count'].mean(),
        'avg_replies': df['reply_count'].mean(),
        'reply_to_tweet_ratio': df['reply_count'].sum() / len(df) if len(df) > 0 else 0,
    }
    return features


def extract_text_features(df):
    """
    Extract text-based features from tweets.
    
    Args:
        df: DataFrame with 'text' column
        
    Returns:
        Dictionary of text features
    """
    features = {
        'avg_text_length': df['text'].str.len().mean(),
        'exclamation_count': df['text'].str.count('!').sum(),
        'question_count': df['text'].str.count(r'\?').sum(),
        'caps_ratio': df['text'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        ).mean(),
    }
    return features


def create_feature_vector(df):
    """
    Create a complete feature vector for prediction.
    
    Args:
        df: DataFrame with all tweet data
        
    Returns:
        Dictionary of all features
    """
    # Extract all feature types
    df = extract_sentiment_features(df)
    engagement_features = extract_engagement_features(df)
    text_features = extract_text_features(df)
    
    # Sentiment aggregations
    sentiment_features = {
        'sentiment_variance': df['vader_compound'].var(),
        'sentiment_mean': df['vader_compound'].mean(),
        'sentiment_std': df['vader_compound'].std(),
        'positive_ratio': (df['vader_compound'] > 0.05).sum() / len(df),
        'negative_ratio': (df['vader_compound'] < -0.05).sum() / len(df),
        'neutral_ratio': ((df['vader_compound'] >= -0.05) & (df['vader_compound'] <= 0.05)).sum() / len(df),
    }
    
    # Combine all features
    all_features = {
        **engagement_features,
        **text_features,
        **sentiment_features,
    }
    
    return all_features
