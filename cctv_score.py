import pandas as pd
import numpy as np

def calculate_cctv_score():
    # 인구 데이터 불러오기 및 전처리
    people = pd.read_csv('data/people.csv')
    people = people.iloc[2:].reset_index(drop=True)
    people = people.drop(columns=['성별(1)'])
    people = people.rename(columns={'자치구별(1)': '자치구', '2024 4/4': '인구수'})
    people['인구수'] = people['인구수'].astype(int)

    # CCTV 데이터
    cctv2024 = pd.read_csv('data/cctv2024.csv', encoding='utf-8')
    cctv2024 = cctv2024.iloc[2:].reset_index(drop=True)

    cctv2024.columns = [
        'Unnamed:0', 'Unnamed:1', '구분', '총계', '2015년 이전', '2016년', '2017년', '2018년',
        '2019년', '2020년', '2021년', '2022년', '2023년', '2024년'
    ]
    cctv2024 = cctv2024[['구분', '총계']].dropna(subset=['구분', '총계'])
    cctv2024['구분'] = cctv2024['구분'].str.replace(' ', '')
    cctv2024 = cctv2024[cctv2024['구분'] != '계'].reset_index(drop=True)
    cctv2024['총계'] = cctv2024['총계'].astype(str).str.replace(',', '').astype(int)
    cctv2024 = cctv2024.rename(columns={'구분': '자치구'})

    # 병합 및 점수 계산
    merged = pd.merge(cctv2024, people, on='자치구')
    merged['cctv_per_10k'] = (merged['총계'] / merged['인구수']) * 10000
    merged['log_cctv'] = np.log1p(merged['cctv_per_10k'])

    merged['cctv점수'] = (
        (merged['log_cctv'] - merged['log_cctv'].min()) /
        (merged['log_cctv'].max() - merged['log_cctv'].min())
    ) * 100
    merged['cctv점수'] = merged['cctv점수'].round(2)

    return merged[['자치구', 'cctv점수']]