# malhaebang-data


### 네이버 부동산 매물 크롤링 

**🔎 목적**

- 서울시 전체 행정구(25개)/동 기준으로 네이버 부동산 원룸/투룸 크롤링하여 각 매물에 대한 상세정보 및 위/경도 좌표를 확보

---

**📍 Tech Stack**

| Component | Technology Used |
| ---- | ---- |
| GUI 기반 크롤링 | Selenium |
| HTML 파싱 | BeautifulSoup |
| DB 저장 | MySQL |
| 위치 좌표 수집 | 내부 API 호출 |

---

**🔁 매물 데이터 수집 파이프라인** `crawling.ipynb`

1. 행정구역 기반 매물 수집
   - 서울시 전체 구/동 행정코드와 명칭 확보
     - 내부 API를 통해 수집 후 `dong_dict.json` 형식으로 구조화
   - 네이버 부동산 페이지에서 서울 → 구 → 동 순서로 클릭하여 접근
     - 동 선택 후, 페이지 끝까지 자동 스크롤하여 모든 매물 리스트 로딩
     - 동마다 매물 수가 달라 전체 매물 수집 완료 후 다음 동/구로 이동
     - Lazy loading 구조 대응을 위한 무한 스크롤 감지 및 종료 조건 처리

2. 매물 상세 정보 추출
   - 각 매물 요소 클릭 → 상세 페이지 진입 (실패 시 `try/except` 및 우회 처리)
   - 주요 항목은 **Selenium + XPath**로 파싱

3. 위치 좌표(위도/경도) 추출
   - 매물 클릭 후 내부 API 호출
   - 응답 JSON에서 `latitude`, `longitude` 필드 파싱
   - 위치 정보는 클릭 이후 로딩되므로 GUI 기반 진입 필수

4. MySQL 적재
   - 수집된 매물 데이터를 정규화된 컬럼에 저장
   - 기존 항목 외에도 `safety_score`, `embedding`, `gpt_description` 필드를 포함한 구조 확보

