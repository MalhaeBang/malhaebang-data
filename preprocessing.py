import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

import mysql.connector
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from config import DB_CONFIG
from sqlalchemy import text

# MySQL 연결 설정
db_config = DB_CONFIG

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)

# 연결
conn = mysql.connector.connect(**db_config)

# 데이터 가져오기
query = "SELECT * FROM house"
df_origin = pd.read_sql(query, conn)

# 데이터 개수 확인
print(f"원본 데이터 개수: {len(df_origin)}")

# 연결 종료
conn.close()

# 데이터 전처리 함수들
def parse_price_column(df, col='price'):
    def parse_price(text):
        try:
            text = str(text).replace(",", "").replace(" ", "")
            if "월세" in text or "단기임대" in text:
                match = re.search(r'(?:월세|단기임대)([\d억]+)[/\\](\d+)', text)
                if match:
                    deposit_raw = match.group(1)
                    if '억' in deposit_raw:
                        parts = deposit_raw.split('억')
                        deposit = int(parts[0]) * 10000
                        if len(parts) > 1 and parts[1].isdigit():
                            deposit += int(parts[1])
                    else:
                        deposit = int(deposit_raw)
                    deposit *= 10000
                    monthly = int(match.group(2)) * 10000
                    return deposit, monthly
            elif "전세" in text:
                match = re.search(r'전세\s*([\d억]+)', text)
                if match:
                    deposit_raw = match.group(1)
                    if '억' in deposit_raw:
                        parts = deposit_raw.split('억')
                        deposit = int(parts[0]) * 10000
                        if len(parts) > 1 and parts[1].isdigit():
                            deposit += int(parts[1])
                    else:
                        deposit = int(deposit_raw)
                    deposit *= 10000
                    return deposit, 0
        except:
            pass
        return None, None

    df[['deposit', 'monthly_rent']] = df[col].apply(lambda x: pd.Series(parse_price(x)))
    return df

def parse_area_column(df, col='area_size'):
    def parse_area(area_text):
        try:
            matches = re.findall(r'([\d.]+)㎡', str(area_text))
            if len(matches) >= 1:
                return float(matches[-1])
        except:
            pass
        return None

    df['전용면적'] = df[col].apply(parse_area)
    df['space'] = df['전용면적']/(3.3058)
    return df

def parse_rooms_column(df, col='rooms_count'):
    def parse_rooms(text):
        try:
            rooms, baths = re.findall(r'(\d+)', str(text))
            return int(rooms), int(baths)
        except:
            return None, None
    df[['rooms_count', 'bath_count']] = df[col].apply(lambda x: pd.Series(parse_rooms(x)))
    return df

def parse_floor_column(df, col='floor'):
    def parse_floor(text):
        try:
            text = str(text).replace(" ", "")
            parts = text.split('/')
            if len(parts) == 2:
                층_raw = parts[0]
                총층_match = re.search(r'(\d+)', parts[1])
                총층 = int(총층_match.group(1)) if 총층_match else None
                해당층_match = re.match(r'(\d+)', 층_raw)
                해당층 = int(해당층_match.group(1)) if 해당층_match else None
                return 해당층, 총층
        except:
            pass
        return None, None
    df[['floor', 'total_floor']] = df[col].apply(lambda x: pd.Series(parse_floor(x)))
    return df

# 전처리 실행
df = df_origin.copy()
print(f"전처리 시작 전 데이터 개수: {len(df)}")

df = parse_price_column(df)
print(f"가격 파싱 후 데이터 개수: {len(df)}")

df = parse_area_column(df)
print(f"면적 파싱 후 데이터 개수: {len(df)}")

df = parse_rooms_column(df)
print(f"방 개수 파싱 후 데이터 개수: {len(df)}")

df = parse_floor_column(df)
print(f"층수 파싱 후 데이터 개수: {len(df)}")

# 추가 전처리
df['space'] = df['space'].fillna(0).round(0).astype(int)
df = df.drop(['전용면적'], axis=1)
print(f"면적 전처리 후 데이터 개수: {len(df)}")

# 컬럼명 수정
df = df.rename(columns={'availabe_from': 'available_from'})

# 데이터 타입 변환
# 1. management_fee
df['management_fee'] = df['management_fee'].astype(str)
df['management_fee'] = df['management_fee'].apply(
    lambda x: np.nan if '정보없음' in x or '정보 없음' in x else (
        int(re.sub(r'[^\d]', '', x)) * 10000 if '만원' in x else pd.to_numeric(re.sub(r'[^\d]', '', x), errors='coerce')
    )
)
df['management_fee'] = pd.to_numeric(df['management_fee'], errors='coerce').astype('Int64')

# 2. agent_comm
df['agent_comm'] = df['agent_comm'].astype(str)
df['agent_comm'] = df['agent_comm'].apply(
    lambda x: np.nan if '정보없음' in x or '정보 없음' in x else (
        int(re.sub(r'[^\d]', '', x)) * 10000 if '만원' in x else pd.to_numeric(re.sub(r'[^\d]', '', x), errors='coerce')
    )
)
df['agent_comm'] = pd.to_numeric(df['agent_comm'], errors='coerce').astype('Int64')

# 3. 숫자형 컬럼
cols_to_int = ['house_num', 'rooms_count', 'bath_count', 'floor', 'total_floor']
for col in cols_to_int:
    df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')

# 4. parking 컬럼
df['parking'] = df['parking'].map({'가능': 1, '불가능': 0, '가능 ': 1, '불가': 0})
df['parking'] = df['parking'].fillna(0).astype(bool)

# 5. 날짜형 컬럼
df['posted_at'] = pd.to_datetime(df['posted_at'], errors='coerce').dt.date
df['built_date'] = df['built_date'].replace('정보 없음', np.nan)
df['built_date'] = pd.to_datetime(df['built_date'], errors='coerce').dt.date

# MySQL에 처리된 데이터 저장
print(f"전체 데이터 개수: {len(df)}")
df.to_sql('cleaned_house', engine, index=False, if_exists='replace')

# 저장된 데이터 개수 확인
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM cleaned_house")).scalar()
    print(f"저장된 데이터 개수: {result}") 