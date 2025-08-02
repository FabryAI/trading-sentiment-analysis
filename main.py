from src.data_loader import load_raw_news
from src.preprocessing import preprocess_dataframe
from src.fetch_news import fetch_news, NewsCategory
from src.model import train_logistic_model, predict_sentiment
from datetime import datetime

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# 🔧 1. Select category
SELECTED_CATEGORY = NewsCategory.STOCKS # Change to .GENERAL or .STOCKS or .CRYPTO as needed

# 🔧 2. Setup output path
output_dir = "src/outputs"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
category_name = SELECTED_CATEGORY.name.lower()
filename = f"{category_name}_sentiment_{timestamp}.png"
output_path = os.path.join(output_dir, filename)

if __name__ == "__main__":
    # 3. Load and preprocess historical dataset
    df = load_raw_news("data/raw/raw_news.csv")
    print("Before preprocessing:")
    print(df.head())

    df = preprocess_dataframe(df)
    print("\nAfter preprocessing:")
    print(df[["text", "clean_text"]].head())

    # 4. Train the model
    model, vectorizer = train_logistic_model(df)

    # 5. Fetch live news
    print(f"\n🔎 Real-time News Feed ({SELECTED_CATEGORY.name}):")
    news_titles = fetch_news(limit=50, category=SELECTED_CATEGORY)

    if not news_titles:
        print("❌ Exiting: No news to analyze.")
        exit()

    # 6. Predict sentiment
    print("\n📊 Predicted Sentiment:")
    predictions = predict_sentiment(model, vectorizer, news_titles)
    for title, sentiment in zip(news_titles, predictions):
        print(f"[{sentiment.upper()}] {title}")

    # 7. Analyze sentiment
    counts = Counter(predictions)
    total = len(predictions)
    percentages = {
        label: round((counts[label] / total) * 100, 1)
        for label in ["positive", "neutral", "negative"]
    }
    dominant = max(percentages, key=percentages.get)

    print("\n📢 Market Sentiment Summary:")
    print(f"> Most common sentiment: {dominant.upper()}")
    print("> Breakdown:")
    for label, pct in percentages.items():
        print(f"  - {label.capitalize()}: {pct}%")

    # 8. Plot and save chart
    df_plot = pd.DataFrame({
        "Sentiment": list(counts.keys()),
        "Count": list(counts.values())
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x="Sentiment", y="Count", data=df_plot,
        palette={"positive": "green", "neutral": "gray", "negative": "red"}
    )

    plt.title(f"Market Sentiment from Live {category_name.capitalize()} News")
    plt.ylabel("Number of Headlines")
    plt.xlabel("Sentiment")
    plt.tight_layout()

    # ✅ Save before show
    plt.savefig(output_path)
    plt.show()

    print(f"\n📁 Sentiment barplot saved to: {output_path}")

