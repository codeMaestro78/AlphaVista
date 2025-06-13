import http.client
import urllib.parse
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MEDIASTACK_API_KEY = '25c464eff5ed7e61cac2c3e557b2821b'  # Replace with your mediastack access key

analyzer = SentimentIntensityAnalyzer()

def fetch_news(symbol, limit=20):
    conn = http.client.HTTPConnection('api.mediastack.com')
    params = urllib.parse.urlencode({
        'access_key': MEDIASTACK_API_KEY,
        'keywords': symbol,
        'categories': '-general,-sports',
        'sort': 'published_desc',
        'limit': limit,
    })
    conn.request('GET', '/v1/news?{}'.format(params))
    res = conn.getresponse()
    data = res.read()
    import json
    articles = json.loads(data.decode('utf-8')).get('data', [])
    news_df = pd.DataFrame([
        {
            'title': a.get('title', ''),
            'description': a.get('description', ''),
            'publishedAt': a.get('published_at', '')[:10],
            'url': a.get('url', ''),
            'source': a.get('source', '')
        } for a in articles
    ])
    return news_df

def compute_sentiment(news_df):
    """Compute sentiment score for each news headline."""
    if news_df.empty:
        return news_df
    # Combine title and description for sentiment
    news_df['text'] = news_df['title'].fillna('') + '. ' + news_df['description'].fillna('')
    news_df['sentiment'] = news_df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return news_df

def daily_sentiment_trend(news_df, window=3):
    """Aggregate sentiment by day."""
    if news_df.empty:
        return pd.DataFrame()
    trend = news_df.groupby('publishedAt')['sentiment'].mean().reset_index()
    trend['publishedAt'] = pd.to_datetime(trend['publishedAt'])
    # Rolling average for smoothing
    trend['rolling_sentiment'] = trend['sentiment'].rolling(window=window, min_periods=1).mean()
    return trend 