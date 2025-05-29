import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from crime_score import calculate_crime_score
from cctv_score_ import calculate_cctv_score
from path_score import calculate_path_score
import sqlite3

def main():
    # Load each score
    crime_df = calculate_crime_score()
    cctv_df = calculate_cctv_score()
    path_df = calculate_path_score()

    # Merge 구 단위 점수들
    final = pd.merge(path_df, cctv_df, on="자치구", how="left")
    final = pd.merge(final, crime_df, on="자치구", how="left")

    # 안전지수 계산
    final["안전지수"] = (
        final["범죄율점수"] * 0.4 +
        final["cctv점수"] * 0.4 +
        final["귀갓길_점수"] * 0.2
    ).round(2)

    # ------------------ 누락 동 보완 ------------------
    with open("dong_dict.json", "r", encoding="utf-8") as f:
        dong_dict = json.load(f)

    all_dongs = [
        {'자치구': gu, '동': dong_name}
        for gu, dongs in dong_dict.items()
        for _, dong_name in dongs
    ]
    full_dong_df = pd.DataFrame(all_dongs)

    existing = set(zip(final['자치구'], final['동']))
    all_set = set(zip(full_dong_df['자치구'], full_dong_df['동']))
    missing = list(all_set - existing)
    missing_df = pd.DataFrame(missing, columns=["자치구", "동"])

    if not missing_df.empty:
        # 구별 점수 범위 계산
        ranges = final.groupby('자치구')[['cctv점수', '범죄율점수', '귀갓길_점수', '안전지수']].agg(['min', 'max'])
        ranges.columns = ['_'.join(col) for col in ranges.columns]
        ranges = ranges.reset_index()

        missing_df = pd.merge(missing_df, ranges, on='자치구', how='left')

        np.random.seed(42)
        for col in ['cctv점수', '범죄율점수', '귀갓길_점수']:
            missing_df[col] = missing_df.apply(
                lambda r: round(np.random.uniform(r[f"{col}_min"], r[f"{col}_max"]), 2), axis=1)

        missing_df['안전지수'] = (
            missing_df['범죄율점수'] * 0.4 +
            missing_df['cctv점수'] * 0.4 +
            missing_df['귀갓길_점수'] * 0.2
        ).round(2)

        # 컬럼 정리 후 추가
        missing_final = missing_df[['자치구', '동', 'cctv점수', '범죄율점수', '귀갓길_점수', '안전지수']]
        final = pd.concat([final, missing_final], ignore_index=True)

    # ------------------ 스케일링 및 등수 ------------------
    def scale_per_gu(group_df):
        scaler = MinMaxScaler(feature_range=(60, 100))
        group_df = group_df.copy()
        group_df["안전지수_스케일링"] = scaler.fit_transform(group_df[["안전지수"]])
        group_df["안전지수_스케일링"] = group_df["안전지수_스케일링"].round(0).astype(int)
        return group_df

    # final = final.groupby("자치구").apply(scale_per_gu).reset_index(drop=True)
    final = (
        final.groupby("자치구", group_keys=False)
        .apply(scale_per_gu)
        .reset_index(drop=True)
    )
    final["구내_등수"] = (
        final.groupby("자치구")["안전지수_스케일링"].rank(ascending=False, method="dense").astype(int)
    )
    final["구내_전체"] = (
        final.groupby("자치구")["안전지수_스케일링"].transform("count").astype(int)
    )
    final["안전점수"] = final["안전지수_스케일링"].astype(str) + "점"
    final["구내_등수표기"] = (
        final["자치구"] + " 내 " +
        final["구내_등수"].astype(str) + "/" +
        final["구내_전체"].astype(str) + "위"
    )

    # print(final[final["자치구"] == "강남구"][["자치구", "동", "안전점수", "구내_등수표기"]])
    cols = [
        "자치구", "동", "cctv점수", "범죄율점수", "귀갓길_점수",
        "안전지수", "안전지수_스케일링", "구내_등수", "구내_전체",
        "안전점수", "구내_등수표기"
    ]
    final = final.sort_values(by=["자치구"]).reset_index(drop=True)
    print(final[cols].head(14).to_string(index=False))

    # safety 등수 데이터 준비
    safety_df = final[["자치구", "동", "구내_등수표기"]].copy()
    safety_df.rename(columns={"자치구": "gu", "동": "dong", "구내_등수표기": "safety_score"}, inplace=True)

    # DB 연결
    conn = sqlite3.connect("test.db")  # 경로는 실제 위치에 맞게 수정
    cursor = conn.cursor()

    # DB에 병합하기
    for _, row in safety_df.iterrows():
        cursor.execute("""
            UPDATE cleaned_house
            SET safety_score = ?
            WHERE gu = ? AND dong = ?
        """, (row["safety_score"], row["gu"], row["dong"]))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
