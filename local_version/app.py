from flask import Flask, render_template, request
import pandas as pd
import random

app = Flask(__name__)

@app.route('/')
def index():
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

    # exclude_urls 전달
    return render_template("index.html", main=main_list, exclude_urls=exclude_urls)

@app.route('/more')
def more():
    df = pd.read_csv("filtered_clustered_news.csv").dropna(subset=["title", "url", "content", "img"])

    # 대표 뉴스에서 사용된 URL 제외
    exclude_urls = request.args.getlist('exclude_urls')

    # 제외한 기사만 남기고 무작위 셔플 후 상위 10개
    remaining_df = df[~df["url"].isin(exclude_urls)]
    sample = remaining_df.sample(frac=1).head(10)

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
    app.run(port=8001, debug=True)
