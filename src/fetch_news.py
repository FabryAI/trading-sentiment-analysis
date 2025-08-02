import feedparser
from typing import List

import requests
from bs4 import BeautifulSoup
from src.categories import NewsCategory

def fetch_yahoo_news(limit=5, category=NewsCategory.GENERAL):
    url = f"https://finance.yahoo.com/{category.value}/"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = [
        a.text.strip()
        for a in soup.select("h3 a") if a.text.strip()
    ]
    return headlines[:limit]

