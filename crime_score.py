import pandas as pd
import numpy as np

def calculate_crime_score():
    people = pd.read_csv('data/people.csv')
    people = people.iloc[2:].reset_index(drop=True)
    people = people.drop(columns=['성별(1)'])
    people = people.rename(columns={'자치구별(1)': '자치구', '2024 4/4': '인구수'})
    people = people.rename(columns={'구분': '자치구'})
    people['인구수'] = people['인구수'].astype(int)

    crime = pd.read_csv('data/crime.csv', encoding='cp949')
    target_types = ['강력범죄', '절도범죄', '폭력범죄', '마약범죄', '풍속범죄']
    crime_filtered = crime[crime['범죄대분류'].isin(target_types)]
    gu_col = [col for col in crime_filtered.columns if col.startswith('서울')]
    gu_sum = crime_filtered[gu_col].sum()

    crime_df = pd.DataFrame({
        '자치구': [col.replace('서울', '').replace(' ', '') for col in gu_col],
        '범죄건수': gu_sum.values
    })

    merged = pd.merge(crime_df, people, on='자치구')
    merged['1만명당범죄율'] = (merged['범죄건수'] / merged['인구수']) * 10000
    max_rate = merged['1만명당범죄율'].max()
    merged['범죄율점수'] = (1 - (merged['1만명당범죄율'] / max_rate)) * 100
    merged['범죄율점수'] = merged['범죄율점수'].round(2)

    return merged[['자치구', '범죄율점수']]