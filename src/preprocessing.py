"""
Text preprocessing utilities for tweets.
"""

import re
import pandas as pd


def clean_text(text):
    """
    Clean tweet text by removing URLs, mentions, hashtags, etc.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text string
    """
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (keep the text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.lower()


def preprocess_dataframe(df):
    """
    Preprocess a dataframe of tweets.
    
    Args:
        df: pandas.DataFrame with 'text' column
        
    Returns:
        DataFrame with added 'cleaned_text' column
    """
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df
