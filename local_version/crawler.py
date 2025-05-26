from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import threading
import time
import re
from langdetect import detect
from tqdm import tqdm
from filtered import getNouns, getVector
from sklearn.cluster import DBSCAN

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--headless")

category_xpath_map = {
    "금융": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[1]/a',
    "증권": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[2]/a',
    "산업/재계": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[3]/a',
    "중기/벤처": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[4]/a',
    "부동산": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[5]/a',
    "글로벌 경제": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[6]/a',
    "생활경제": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[7]/a',
    "경제 일반": '/html/body/div/div[2]/div[2]/div[1]/div/ul/li[8]/a'
}

class UrlCrawling:
    def __init__(self):
        self.category_names = list(category_xpath_map.keys())
        self.url_df_list = [None] * len(self.category_names)
        self.lock = threading.Lock()

    def getUrl(self, idx):
        a_tag_list, urls, category_list = [], [], []
        browser = webdriver.Chrome(options=options)

        browser.get("https://news.naver.com/")
        time.sleep(1)
        wait = WebDriverWait(browser, 10)
        wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/section/header/div[2]/div/div/div/div/div/ul/li[3]/a'))).click()
        time.sleep(1)
        wait.until(EC.element_to_be_clickable((By.XPATH, list(category_xpath_map.values())[idx]))).click()
        time.sleep(1)

        try:
            for _ in range(2):
                browser.find_element(By.CLASS_NAME, "_CONTENT_LIST_LOAD_MORE_BUTTON").click()
                time.sleep(1)
        except:
            pass

        soup = BeautifulSoup(browser.page_source, "html.parser")
        a_tag_list.extend(soup.select(".section_latest ._TEMPLATE .sa_thumb_link"))
        #  a_tag_list = a_tag_list[:10]  # test

        for a in a_tag_list:
            urls.append(a["href"])
            category_list.append(self.category_names[idx])

        url_df = pd.DataFrame({'category': category_list, 'url': urls})
        with self.lock:
            self.url_df_list[idx] = url_df
        browser.quit()

class ContentCrawling:
    def __init__(self):
        self.category_names = list(category_xpath_map.keys())
        self.title = [[] for _ in range(len(self.category_names))]
        self.content = [[] for _ in range(len(self.category_names))]
        self.img = [[] for _ in range(len(self.category_names))]
        self.lock = threading.Lock()

    def getContent(self, url_list, idx):
        title_list, content_list, img_list = [], [], []
        browser = webdriver.Chrome(options=options)
        for url in tqdm(url_list, desc=f"{self.category_names[idx]} 수집 진행", position=0):
            browser.get(url)
            time.sleep(0.5)
            soup = BeautifulSoup(browser.page_source, "html.parser")
            try:
                title_tag = soup.select("#title_area span")[0]
                content_raw = soup.find_all(attrs={"id": "dic_area"})
                if not content_raw:
                    continue
                self.getImg(soup, img_list)
                title_list.append(title_tag)
                content_list.append(self.removeTag(content_raw))
            except:
                continue
        with self.lock:
            for i in range(len(title_list)):
                try:
                    self.title[idx].append(title_list[i].text.strip())
                    self.content[idx].append(cleanContent(content_list[i][0].text.strip()))
                    self.img[idx].append(img_list[i])
                except:
                    continue
        browser.quit()

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
        news_df.dropna(subset=['content'], inplace=True)
        startRemove(news_df)
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

# ---------- 전처리 ----------
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
    df.drop(df[df['title'].str.contains('사진|포토|영상|움짤|헤드라인|라이브|정치쇼')].index, inplace=True)
    df.drop(df[df['content'].str.contains('방송 :|방송:|진행 :|진행:|출연 :|출연:|앵커|[앵커]')].index, inplace=True)

def startRemove(df):
    EtcNews(df)
    shortNews(df)
    englishNews(df)
    df.drop_duplicates(subset=["url"], inplace=True)

# ---------- 실행용 ----------
def urlThread(url_crawler):
    for idx in range(len(category_xpath_map)):
        url_crawler.getUrl(idx)

def contentThread(url_crawler, content_crawler):
    url_list, all_url_list, category_list = [], [], []
    for i in range(len(category_xpath_map)):
        if url_crawler.url_df_list[i] is None:
            continue
        url_list.append(list(url_crawler.url_df_list[i]['url']))
        all_url_list.extend(url_crawler.url_df_list[i]['url'])
        category_list.extend(url_crawler.url_df_list[i]['category'])

    for i in range(len(url_list)):
        content_crawler.getContent(url_list[i], i)

    return content_crawler.makeDataFrame(all_url_list, category_list)

def startCrawling():
    url_crawler = UrlCrawling()
    content_crawler = ContentCrawling()
    urlThread(url_crawler)
    news_df = contentThread(url_crawler, content_crawler)
    print(f"✅ 크롤링 완료, 기사 수: {len(news_df)}")
    
    getNouns(news_df)
    vector_list = getVector(news_df)

    return news_df
