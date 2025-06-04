### 뉴스 크롤링

**🔎 목적**

- 네이버 뉴스 경제 카테고리 전체에서 최신 기사를 세부 카테고리별로 수집하고 제목/본문/썸네일/URL을 확보하여 사용자에게 오늘의 이슈 전달

---

**📍 Tech Stack**

| Component | Technology Used |
| --- | ----- |
| Automated Crawling | Selenium, BeautifulSoup, webdriver-manager, tqdm |
| Data Processing | Pandas, Numpy, langdetect, scikit-learn (TF-IDF, KMeans) |
| Text Analysis | KoNLPy (Komoran) |
| Web Server | Flask |
| Template Rendering | Jinja2 |
| Front-end | HTML/CSS |

---

**🔁 파일별 상세 기능**
1. 뉴스 수집 자동화 파이프라인 crawler.py
      - 카테고리 접근 자동화 (XPATH 매핑)
      - 기사 url 목록 수집
      - 기사 본문 및 이미지 수집
      - 텍스트 정제 및 구조화
      - 병렬 실행 흐름
2. 클러스터링 및 분류/필터링 파이프라인 filtered.py
      - 명사 추출 및 벡터화
      - DBSCAN 기반 군집화
      - 클러스터 요약 및 키워드 기반 주제 분류
4. 실행 컨트롤러 main.py
      - 전체 흐름 제어
5. 뉴스 요약 웹 서비스 구현 app.py
      - 대표 뉴스 3건 무작위 제공 (/ 라우트)
      - 기사 10건 무작위 제공 (/more 라우트)
      - hover시 tooltip 표시 및 이미지가 없는 경우 기본 이미지 대체
