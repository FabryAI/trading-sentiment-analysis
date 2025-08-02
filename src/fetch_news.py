import feedparser
from enum import Enum
from typing import List


class NewsCategory(Enum):
    GENERAL = "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"
    STOCKS = "https://www.investing.com/rss/news_25.rss"
    CRYPTO = "https://cointelegraph.com/rss"


def fetch_news(limit=10, category: NewsCategory = NewsCategory.GENERAL) -> List[str]:
    feed_url = category.value
    feed = feedparser.parse(feed_url)

    headlines = [entry.title.strip() for entry in feed.entries[:limit]]

    if not headlines:
        print("⚠️ No headlines fetched. Check the RSS URL or internet connection.")

    return headlines
