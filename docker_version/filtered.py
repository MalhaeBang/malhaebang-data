# filtered.py
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Komoran
from sklearn.cluster import DBSCAN
import pandas as pd
import re
from collections import Counter

# ---------- 명사 추출 ----------
def getNouns(news_df):
    komoran = Komoran()
    nouns_list = []

    for idx, content in enumerate(news_df["content"]):
        if not isinstance(content, str) or not content.strip():
            nouns_list.append([])
            continue

        try:
            sentences = re.split(r'[.!?\\n]+', content)
            all_nouns = []
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    nouns = komoran.nouns(sent)
                    all_nouns.extend(nouns)
            nouns_list.append(all_nouns)
        except Exception as e:
            print(f"[{idx}] Komoran 오류: {repr(content[:50])} -> {e}")
            nouns_list.append([])

    news_df["nouns"] = nouns_list

# ---------- 벡터화 ----------
def getVector(news_df):
    category_names = news_df["category"].unique()
    vector_list = []

    for category in category_names:
        text = [" ".join(noun) for noun in news_df['nouns'][news_df['category'] == category]]
        text = [t for t in text if t.strip()]

        if not text or len(text) < 2:
            continue

        tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 5))
        tfidf_vectorizer.fit(text)
        vector = tfidf_vectorizer.transform(text).toarray()
        vector_list.append(vector)

    return vector_list

# ---------- 군집 처리 ----------
def addClusterNumber(news_df, vector_list):
    cluster_number_list = []
    for vector in vector_list:
        model = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        result = model.fit_predict(vector)
        cluster_number_list.extend(result)

    news_df['cluster_number'] = cluster_number_list

# ---------- 군집 대표 기사 추출 ----------
def get_deduplicated_copy(df):
    df['cluster_number'] = df['cluster_number'].fillna(-1).astype(int)
    clustered = df[df['cluster_number'] != -1]
    clustered = clustered.sort_values('title').drop_duplicates(subset=['category', 'cluster_number'], keep='first')
    noise = df[df['cluster_number'] == -1]
    return pd.concat([clustered, noise], ignore_index=True)

# ---------- 군집 요약 ----------
def getClusteredNews(news_df):
    category_names = news_df['category'].unique()
    cluster_counts_df = pd.DataFrame(columns=["category", "cluster_number", "cluster_count"])

    for category in category_names:
        tmp = news_df[news_df['category'] == category]['cluster_number'].value_counts().reset_index()
        tmp.columns = ['cluster_number', 'cluster_count']
        tmp['category'] = [category] * len(tmp)
        cluster_counts_df = pd.concat([cluster_counts_df, tmp], ignore_index=True)

    cluster_counts_df = (
        cluster_counts_df
        .sort_values(['category', 'cluster_count'], ascending=[True, False])
        .groupby(['category'])
        .head(10)
        .reset_index(drop=True)
    )

    return cluster_counts_df

# ---------- 키워드 기반 분류 ----------
category_col = '통합 분류1'
keyword_cols = ['키워드', '특성추출']

def load_category_keywords(data_path='data.csv', top_n=30):
    data = pd.read_csv(data_path)
    category_keywords = {}
    for category in data[category_col].dropna().unique():
        cat_df = data[data[category_col] == category]
        keywords = []
        for col in keyword_cols:
            for row in cat_df[col].astype(str):
                keywords.extend([token.strip() for token in row.split(',') if token.strip()])
        most_common = [kw for kw, _ in Counter(keywords).most_common(top_n)]
        category_keywords[category] = set(most_common)
    return category_keywords

def get_related_categories(nouns, category_keywords):
    return [cat for cat, kws in category_keywords.items() if set(nouns) & kws]

# ---------- 실행 파이프라인 ----------
def filter_and_save_news(
    news_path='대표뉴스.csv',
    data_path='data.csv',
    cluster_output_path='clustered.csv',
    final_output_path='filtered_clustered_news.csv',
    exclude_keywords=None,
    top_n=30
):
    news_df = pd.read_csv(news_path)
    getNouns(news_df)
    vector_list = getVector(news_df)
    addClusterNumber(news_df, vector_list)
    dedup_df = get_deduplicated_copy(news_df)
    dedup_df.to_csv(cluster_output_path, index=False)
    print(f"✅ 군집 대표 기사 {len(dedup_df)}건 저장됨 → {cluster_output_path}")

    category_keywords = load_category_keywords(data_path, top_n=top_n)
    dedup_df['related_categories'] = dedup_df['nouns'].apply(lambda n: get_related_categories(n, category_keywords))
    filtered_df = dedup_df[dedup_df['related_categories'].apply(lambda cats: len(cats) > 0)]

    if exclude_keywords:
        filtered_df = filtered_df[
            filtered_df['related_categories'].apply(
                lambda cats: len([c for c in cats if not any(x in c for x in exclude_keywords)]) > 0
            )
        ]

    filtered_df.to_csv(final_output_path, index=False)
    print(f"✅ 최종 필터링된 대표 기사 {len(filtered_df)}건 저장됨 → {final_output_path}")
    return filtered_df

# ---------- 전체 실행 진입점 ----------
def startClustering(news_df):
    getNouns(news_df)
    vector_list = getVector(news_df)
    addClusterNumber(news_df, vector_list)
    cluster_counts_df = getClusteredNews(news_df)
    return cluster_counts_df
