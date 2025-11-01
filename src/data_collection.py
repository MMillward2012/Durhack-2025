"""
Data collection module for tweet polarization predictor.

This module handles fetching tweets on a given topic.
"""

import tweepy
import pandas as pd


def setup_twitter_api(bearer_token):
    """
    Setup Twitter API v2 client.
    
    Args:
        bearer_token: Twitter API Bearer Token
        
    Returns:
        tweepy.Client object
    """
    client = tweepy.Client(bearer_token=bearer_token)
    return client


def fetch_tweets(client, query, max_results=100):
    """
    Fetch tweets based on a search query.
    
    Args:
        client: tweepy.Client object
        query: Search query string
        max_results: Number of tweets to fetch (default: 100)
        
    Returns:
        pandas.DataFrame with tweet data
    """
    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=['created_at', 'public_metrics', 'author_id']
    )
    
    if not tweets.data:
        return pd.DataFrame()
    
    tweet_data = []
    for tweet in tweets.data:
        tweet_data.append({
            'text': tweet.text,
            'created_at': tweet.created_at,
            'retweet_count': tweet.public_metrics['retweet_count'],
            'like_count': tweet.public_metrics['like_count'],
            'reply_count': tweet.public_metrics['reply_count'],
        })
    
    return pd.DataFrame(tweet_data)


def load_sample_data(filepath):
    """
    Load sample tweet data from CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pandas.DataFrame
    """
    return pd.read_csv(filepath)
