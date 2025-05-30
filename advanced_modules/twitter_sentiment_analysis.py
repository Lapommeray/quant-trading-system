import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TwitterSentimentAnalyzer:
    """
    Twitter NLP Sentiment Analysis for trading signals
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_history = []
        
    def analyze_tweet(self, tweet_text):
        """
        Analyze sentiment of a single tweet
        """
        sentiment = self.analyzer.polarity_scores(tweet_text)
        return sentiment
    
    def analyze_tweets(self, tweets):
        """
        Analyze sentiment of multiple tweets
        """
        sentiments = [self.analyze_tweet(tweet) for tweet in tweets]
        self.sentiment_history.extend(sentiments)
        
        compound_scores = [s['compound'] for s in sentiments]
        
        return {
            'mean_sentiment': np.mean(compound_scores),
            'sentiment_std': np.std(compound_scores),
            'bullish_ratio': sum(1 for s in compound_scores if s > 0.05) / len(compound_scores),
            'bearish_ratio': sum(1 for s in compound_scores if s < -0.05) / len(compound_scores),
            'neutral_ratio': sum(1 for s in compound_scores if abs(s) <= 0.05) / len(compound_scores),
            'raw_sentiments': sentiments
        }
    
    def get_trading_signal(self, tweets, threshold=0.2):
        """
        Convert sentiment to trading signal
        """
        analysis = self.analyze_tweets(tweets)
        
        if analysis['mean_sentiment'] > threshold:
            return 1  # Buy signal
        elif analysis['mean_sentiment'] < -threshold:
            return -1  # Sell signal
        else:
            return 0  # Neutral
