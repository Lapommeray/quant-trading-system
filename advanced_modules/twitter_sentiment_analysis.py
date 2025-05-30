import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

class TwitterSentimentAnalyzer:
    """
    Twitter NLP sentiment analysis for alternative data integration
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sentiment_history = {}
        
    def analyze_tweet(self, tweet_text):
        """
        Analyze sentiment of a single tweet
        """
        if not tweet_text or not isinstance(tweet_text, str):
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0
            }
            
        try:
            sentiment = self.analyzer.polarity_scores(tweet_text)
            return sentiment
        except Exception as e:
            self.logger.error(f"Error analyzing tweet: {str(e)}")
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0
            }
            
    def analyze_tweets(self, tweets_list):
        """
        Analyze sentiment of multiple tweets
        """
        if not tweets_list:
            return []
            
        sentiments = []
        for tweet in tweets_list:
            if isinstance(tweet, dict) and 'text' in tweet:
                sentiment = self.analyze_tweet(tweet['text'])
                sentiments.append({
                    'tweet_id': tweet.get('id', ''),
                    'created_at': tweet.get('created_at', ''),
                    'sentiment': sentiment
                })
            elif isinstance(tweet, str):
                sentiment = self.analyze_tweet(tweet)
                sentiments.append({
                    'tweet_id': '',
                    'created_at': '',
                    'sentiment': sentiment
                })
                
        return sentiments
        
    def get_asset_sentiment(self, asset, tweets_list):
        """
        Get sentiment for a specific asset from tweets
        """
        if not tweets_list:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0
            }
            
        asset_tweets = []
        for tweet in tweets_list:
            if isinstance(tweet, dict) and 'text' in tweet:
                text = tweet['text'].lower()
                if asset.lower() in text:
                    asset_tweets.append(tweet)
            elif isinstance(tweet, str):
                if asset.lower() in tweet.lower():
                    asset_tweets.append({'text': tweet})
                    
        if not asset_tweets:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0
            }
            
        sentiments = self.analyze_tweets(asset_tweets)
        
        compound_sum = sum(s['sentiment']['compound'] for s in sentiments)
        pos_sum = sum(s['sentiment']['pos'] for s in sentiments)
        neu_sum = sum(s['sentiment']['neu'] for s in sentiments)
        neg_sum = sum(s['sentiment']['neg'] for s in sentiments)
        
        count = len(sentiments)
        
        avg_sentiment = {
            'compound': compound_sum / count if count > 0 else 0.0,
            'pos': pos_sum / count if count > 0 else 0.0,
            'neu': neu_sum / count if count > 0 else 0.0,
            'neg': neg_sum / count if count > 0 else 0.0
        }
        
        self.sentiment_history[asset] = self.sentiment_history.get(asset, [])
        self.sentiment_history[asset].append(avg_sentiment)
        
        return avg_sentiment
        
    def get_sentiment_trend(self, asset, window=5):
        """
        Get sentiment trend for an asset
        """
        if asset not in self.sentiment_history or len(self.sentiment_history[asset]) < 2:
            return 0.0
            
        history = self.sentiment_history[asset]
        window = min(window, len(history))
        
        recent = history[-window:]
        
        if len(recent) < 2:
            return 0.0
            
        first_sentiment = recent[0]['compound']
        last_sentiment = recent[-1]['compound']
        
        trend = last_sentiment - first_sentiment
        return trend
        
    def get_sentiment_signal(self, asset):
        """
        Convert sentiment to trading signal
        """
        if asset not in self.sentiment_history or not self.sentiment_history[asset]:
            return 0.0
            
        latest_sentiment = self.sentiment_history[asset][-1]['compound']
        trend = self.get_sentiment_trend(asset)
        
        signal = latest_sentiment * 0.7 + trend * 0.3
        
        signal = max(min(signal, 1.0), -1.0)
        
        return signal
