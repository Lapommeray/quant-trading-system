
def get_sentiment_gap(price_trend, reddit_sentiment):
    if price_trend > 0 and reddit_sentiment < -0.7:
        return "WHALE_MANIPULATION"
    elif price_trend < 0 and reddit_sentiment > 0.8:
        return "PUMP_TRAP"
    return "CLEAR"
