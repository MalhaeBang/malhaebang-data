# crawler.py (URL 방식)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import traceback
import threading
import time
import re
from langdetect import detect
from tqdm import tqdm
from filtered import getNouns, getVector
from sklearn.cluster import DBSCAN
from selenium.webdriver.common.by import By

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

category_links = {
    "금융": "259", "증권": "258", "산업/재계": "261", "중기/벤처": "771",
    "부동산": "260", "글로벌 경제": "262", "생활경제": "310", "경제 일반": "263"
}

class UrlCrawling:
    def __init__(self):
        self.category_names = list(category_links.keys())
        self.url_df_list = [None] * len(self.category_names)
        self.lock = threading.Lock()

    def getUrl(self, idx):
        a_tag_list, urls, category_list = [], [], []

        category_name = self.category_names[idx]
        target_code = category_links[category_name]
        url = f"https://news.naver.com/breakingnews/section/101/{target_code}"

        browser = webdriver.Chrome(options=options)
        wait = WebDriverWait(browser, 10)

        try:
            browser.get(url)

            # 기사 리스트 로딩 대기
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".sa_thumb_link")))
            time.sleep(1)

            # '더보기' 버튼 2회 클릭
            try:
                for _ in range(2):
                    browser.find_element(By.CLASS_NAME, "_CONTENT_LIST_LOAD_MORE_BUTTON").click()
                    time.sleep(1)
            except:
                pass

            # 기사 목록 수집
            soup = BeautifulSoup(browser.page_source, "html.parser")
            a_tag_list.extend(soup.select(".section_latest ._TEMPLATE .sa_thumb_link"))

            safe_name = category_name.replace("/", "_")  # 슬래시(/)가 포함된 카테고리 이름 -> Linux에서 디렉터리로 해석해서 에러 발생
            with open(f"/tmp/debug_newslist_{safe_name}.html", "w", encoding="utf-8") as f:
                f.write(browser.page_source)

            for a in a_tag_list[:10]:
                href = a.get("href")
                if href and href.startswith("https://n.news.naver.com/"):
                    urls.append(href)
                    category_list.append(category_name)

            url_df = pd.DataFrame({'category': category_list, 'url': urls})
            with self.lock:
                self.url_df_list[idx] = url_df

        except Exception as e:
            print(f"❌ getUrl({category_name}) 전체 실패:", e)
            traceback.print_exc()

        finally:
            browser.quit()

class ContentCrawling:
    def __init__(self):
        self.category_names = list(category_links.keys())
        self.title = [[] for _ in range(len(self.category_names))]
        self.content = [[] for _ in range(len(self.category_names))]
        self.img = [[] for _ in range(len(self.category_names))]
        self.lock = threading.Lock()

    def getContent(self, url_list, idx):
        title_list, content_list, img_list = [], [], []
        browser = webdriver.Chrome(options=options)

        for url in tqdm(url_list, desc=f"{self.category_names[idx]} 수집 진행", position=0):
            try:
                browser.get(url)
                time.sleep(0.5)
                soup = BeautifulSoup(browser.page_source, "html.parser")

                title_tag = soup.select("#title_area span")
                content_raw = soup.find_all(attrs={"id": "dic_area"})

                if not title_tag or not content_raw:
                    print(f"[WARNING] Missing content at {url}")
                    continue

                title_text = title_tag[0].text.strip()
                content_text = content_raw[0].text.strip()

                if not title_text or not content_text:
                    print(f"[WARNING] Empty title/content at {url}")
                    continue

                img = self.getImg(soup)

                title_list.append(title_text)
                content_list.append(content_text)
                img_list.append(img)

            except Exception as e:
                print(f"[ERROR] Exception at {url}: {e}")
                continue

        with self.lock:
            self.title[idx].extend(title_list)
            self.content[idx].extend(content_list)
            self.img[idx].extend(img_list)

        browser.quit()

    def getImg(self, soup):
        img_tag = soup.select(".end_photo_org img")
        if img_tag:
            img_src_list = [img['src'] for img in img_tag if '.gif' not in img['src']][:10]
            return ",".join(img_src_list)
        return ""

    def makeDataFrame(self, all_url, category):
        title, content, img = [], [], []
        for i in self.title: title.extend(i)
        for i in self.content: content.extend(i)
        for i in self.img: img.extend(i)

        news_df = pd.DataFrame({
            "category": pd.Series(category),
            "title": pd.Series(title),
            "content": pd.Series(content),
            "img": pd.Series(img),
            "url": pd.Series(all_url)
        })

        # print(f"\n[DEBUG] 추출된 기사 수: {news_df['content'].notna().sum()}")
        news_df.dropna(subset=['content'], inplace=True)
        return news_df

def getImg(self, soup, img_list):
        img_tag = soup.select(".end_photo_org img")
        if img_tag:
            img_src_list = []
            for img in img_tag:
                if len(img_src_list) <= 10 and '.gif' not in img['src']:
                    img_src_list.append(img['src'])
            img_list.append(",".join(img_src_list))
        else:
            img_list.append("")

def removeTag(self, content):
    targets = ["strong", "small", "table", "b", {"class": "end_photo_org"}, {"class": "vod_player_wrap"},
                {"id": "video_area"}, {"name": "iframe"}, {"class": "image"}, {"class": "vod_area"},
                {"class": "artical-btm"}, {"class": "caption"}, {"class": "source"},
                {"class": "byline"}, {"class": "reporter_area"}, {"class": "copyright"},
                {"class": "categorize"}, {"class": "promotion"}]
    for target in targets:
        while content[0].find(target):
            content[0].find(target).decompose()
    return content

def cleanContent(text):
    text = re.sub('\([^)]*\)', '', text)
    text = re.sub('\[[^\]]+\]', '', text)
    text = re.sub('([^\s]*\s기자)', '', text)
    text = re.sub('([^\s]*\\온라인 기자)', '', text)
    text = re.sub('([^\s]*\s기상캐스터)', '', text)
    text = re.sub('포토', '', text)
    text = re.sub('\S+@[a-z.]+', '', text)
    text = re.sub('[“”]', '"', text)
    text = re.sub('[‘’]', "'", text)
    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('다\.(?=(?:[^"]*\"[^"]*\")*[^"]*$)', '다.\n', text)
    text = re.sub('[\t\xa0]', '', text)
    text = re.sub('[ㄱ-ㅎㅏ-ㅣ]+', '', text)
    text = re.sub('[=+#/^$@*※&ㆍ!』\\|\[\]<>…》■□ㅁ◆◇▶◀▷◁△▽▲▼○●━]', '', text)
    return text

def shortNews(df):
    df.drop(df[df['content'].apply(lambda x: len(x.split("다."))) <= 4].index, inplace=True)
    df.drop(df[df['content'].apply(len) <= 300].index, inplace=True)

def isEnglish(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def englishNews(df):
    df = df[~df['content'].apply(isEnglish)]

def EtcNews(df):
    df.drop(df[df['title'].astype(str).str.contains('사진|포토|영상|움짤|헤드라인|라이브|정치쇼')].index, inplace=True)
    df.drop(df[df['content'].astype(str).str.contains('방송 :|방송:|진행 :|진행:|출연 :|출연:|앵커|[앵커]')].index, inplace=True)

def startRemove(df):
    EtcNews(df)
    shortNews(df)
    englishNews(df)
    df.drop_duplicates(subset=["url"], inplace=True)

def urlThread(url_crawler):
    # for idx in range(len(category_urls)):
    for idx in range(len(category_links)):
        url_crawler.getUrl(idx)

def contentThread(url_crawler, content_crawler):
    url_list, all_url_list, category_list = [], [], []

    for i in range(len(category_links)):
        if url_crawler.url_df_list[i] is None:
            continue
        url_list.append(list(url_crawler.url_df_list[i]['url']))
        all_url_list.extend(url_crawler.url_df_list[i]['url'])
        category_list.extend(url_crawler.url_df_list[i]['category'])

    for i in range(len(url_list)):
        content_crawler.getContent(url_list[i], i)

    return content_crawler.makeDataFrame(all_url_list, category_list)

def startCrawling():
    try:
        url_crawler = UrlCrawling()
        content_crawler = ContentCrawling()
        urlThread(url_crawler)
        news_df = contentThread(url_crawler, content_crawler)
        print(f"✅ 크롤링 완료, 기사 수: {len(news_df)}")

        getNouns(news_df)
        vector_list = getVector(news_df)

        # print("[DEBUG] news_df 길이:", len(news_df))
        # print("[DEBUG] 벡터 길이:", len(vector_list))

    except Exception as e:
        print("❌ 크롤링 오류:", e)
        traceback.print_exc()
        return None

    return news_df

