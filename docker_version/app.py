# app.py
from flask import Flask, render_template, request
import pandas as pd
import random
import os

app = Flask(__name__)

@app.route('/')
def index():
    if not os.path.exists("filtered_clustered_news.csv"):
        return "❌ filtered_clustered_news.csv 파일이 없습니다.", 500

    df = pd.read_csv("filtered_clustered_news.csv").dropna(subset=["title", "url", "content", "img"])
    main_news = df.sample(min(3, len(df)), random_state=random.randint(1, 999999))
    exclude_urls = main_news["url"].tolist()

    main_list = []
    for _, row in main_news.iterrows():
        summary = ' '.join(row["content"].replace('\n', ' ').split())
        summary = summary[:70] + "…" if len(summary) > 70 else summary
        main_list.append({
            "title": row["title"],
            "url": row["url"],
            "img": str(row["img"]).split(",")[0] if pd.notna(row["img"]) else "/static/default.jpg",
            "summary": summary
        })

    return render_template("index.html", main=main_list, exclude_urls=exclude_urls)

@app.route('/more')
def more():
    if not os.path.exists("filtered_clustered_news.csv"):
        return "❌ filtered_clustered_news.csv 파일이 없습니다.", 500

    df = pd.read_csv("filtered_clustered_news.csv").dropna(subset=["title", "url", "content", "img"])
    exclude_urls = request.args.getlist('exclude_urls')
    remaining_df = df[~df["url"].isin(exclude_urls)]

    sample = remaining_df.sample(min(10, len(remaining_df)), random_state=random.randint(1, 999999))

    more_list = []
    for _, row in sample.iterrows():
        summary = ' '.join(row["content"].replace('\n', ' ').split())
        summary = summary[:80] + "..." if len(summary) > 80 else summary
        more_list.append({
            "title": row["title"],
            "url": row["url"],
            "img": row["img"].split(",")[0] if row["img"] else "",
            "summary": summary
        })

    return render_template("more.html", news_list=more_list)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001, debug=True)