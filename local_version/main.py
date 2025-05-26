from crawler import startCrawling
from filtered import filter_and_save_news as cluster_news
import pandas as pd

# if __name__ == "__main__":
#     df = startCrawling()
#     df.to_csv("대표뉴스.csv", index=False)
#     print("✅ 대표 뉴스 저장 완료 → 대표뉴스.csv")

def main():
    df = None
    try:
        df = startCrawling()
        df.to_csv("대표뉴스.csv", index=False)
        print("✅ 대표 뉴스 저장 완료 → 대표뉴스.csv")
    except Exception as e:
        print(f"❌ 크롤링 오류: {e}")

    if df is not None:
        try:
            # 군집 + 대표 기사 추출 + 필터까지 한 번에
            cluster_news(
                news_path="대표뉴스.csv",
                data_path="data.csv",
                cluster_output_path="clustered.csv",
                final_output_path="filtered_clustered_news.csv"
            )
            print("✅ 필터링 및 군집 완료 → filtered_clustered_news.csv")
        except Exception as e:
            print(f"❌ 군집+필터링 오류: {e}")
    else:
        print("⚠️ 크롤링 실패로 군집 및 필터링 생략")

if __name__ == "__main__":
    main()