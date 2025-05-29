import pandas as pd
import numpy as np
import json

def calculate_path_score():
    # 귀갓길 시설 데이터
    path = pd.read_csv('data/path.csv', encoding='cp949')
    service = pd.read_csv('data/service.csv', encoding='cp949')

    # 결측치 있는 컬럼 제거
    path = path.drop(columns=['안심귀갓길 안내표지판', '안심귀갓길 서비스 안내판', '기타 시설물', '비고', '데이터기준일자'])
    
    # 결측값 처리
    path[['안심벨', 'cctv', '112 위치신고 안내판']] = path[['안심벨', 'cctv', '112 위치신고 안내판']].fillna(0)

    # 가중치 설정
    weight = {
        'cctv': 3.5,
        '보안등': 3.0,
        '안심벨': 2.0,
        '112 위치신고 안내판': 1.5
    }

    # 점수 계산
    path['총점'] = (
        path['cctv'] * weight['cctv'] +
        path['보안등'] * weight['보안등'] +
        path['안심벨'] * weight['안심벨'] +
        path['112 위치신고 안내판'] * weight['112 위치신고 안내판']
    )

    # 동별 집계
    dong_score = (
        path.groupby(['시군구명', '읍면동명'])['총점']
        .sum()
        .reset_index()
    )

    # 정규화 (log 기준)
    dong_score['귀갓길_점수'] = (
        (np.log1p(dong_score['총점']) - np.log1p(dong_score['총점']).min()) /
        (np.log1p(dong_score['총점']).max() - np.log1p(dong_score['총점']).min())
    ) * 100
    dong_score['귀갓길_점수'] = dong_score['귀갓길_점수'].round(2)

    # 자치구/동 컬럼 정제
    dong_score['자치구'] = dong_score['시군구명'].str.replace('서울특별시 ', '')
    dong_score['동'] = dong_score['읍면동명']
    
    return dong_score[['자치구', '동', '귀갓길_점수']]