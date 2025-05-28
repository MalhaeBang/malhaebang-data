from flask import Flask, request, jsonify, render_template
import sqlite3
import numpy as np
import pandas as pd
import faiss
import json
import os

app = Flask(__name__, template_folder="templates")

conn = sqlite3.connect("house_with_embedding.db")
df = pd.read_sql("SELECT * FROM house", conn)

# DB 및 Faiss 인덱스 로딩
conn = sqlite3.connect("house_with_embedding.db", check_same_thread=False)
cursor = conn.cursor()
index = faiss.read_index("faiss_index.faiss")

def get_embedding_from_db(house_num):
    cursor.execute("SELECT final_embedding FROM house WHERE house_num = ?", (house_num,))
    row = cursor.fetchone()
    if row is None:
        return None
    return np.array(json.loads(row[0]), dtype='float32')

def get_house_info_by_ids(house_nums):
    placeholders = ','.join(['?'] * len(house_nums))
    query = f"""
        SELECT house_num, title, address, gu, dong, price, area_size,
               space, img_url, management_fee,
               rooms_count, bath_count, floor, total_floor, house_feature
        FROM house
        WHERE house_num IN ({placeholders})
    """
    cursor.execute(query, house_nums)
    rows = cursor.fetchall()
    columns = ['house_num', 'title', 'address', 'gu', 'dong', 'price', 'area_size', 'space', 'img_url', 'management_fee', 'rooms_count', 'bath_count', 'floor', 'total_floor', 'house_feature']
    df = pd.DataFrame(rows, columns=columns)
    return df

@app.route("/recommend", methods=["GET"])
def recommend():
    house_num = request.args.get("house_num", type=int)
    top_k = request.args.get("top_k", default=10, type=int)

    if house_num is None:
        return jsonify({"error": "house_num parameter required"}), 400

    query_vec = get_embedding_from_db(house_num)
    if query_vec is None:
        return jsonify({"error": f"house_num {house_num} not found"}), 404

    query_vec = query_vec.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, top_k)

    similar_ids = [int(i) for i in I[0]]
    df = get_house_info_by_ids(similar_ids)

    df['similarity'] = D[0][:len(df)]
    df = df[df['similarity'] < 0.9999]
    df = df.drop_duplicates(subset=['title', 'address', 'gu', 'dong', 'price', 'space'])

    return jsonify({"recommendations": df.to_dict(orient="records")})

@app.route("/list")
def house_list():
    query = request.args.get("q", "").strip()
    
    if query:
        cursor.execute("""
            SELECT house_num, title, address, gu, dong, price, area_size,
               space, img_url, management_fee,
               rooms_count, bath_count, floor, total_floor, house_feature
            FROM house
            WHERE title LIKE ?
            ORDER BY RANDOM() LIMIT 20
        """, (f"%{query}%",))
    else:
        cursor.execute("""
            SELECT house_num, title, address, gu, dong, price, area_size,
               space, img_url, management_fee,
               rooms_count, bath_count, floor, total_floor, house_feature
            FROM house
            ORDER BY RANDOM() LIMIT 20
        """)
    
    rows = cursor.fetchall()
    columns = ['house_num', 'title', 'address', 'gu', 'dong', 'price', 'area_size', 'space', 'img_url', 'management_fee',
    'rooms_count', 'bath_count', 'floor', 'total_floor', 'house_feature']
    df = pd.DataFrame(rows, columns=columns)

    def safe_extract_thumbnail(x):
        try:
            parsed = json.loads(x) if isinstance(x, str) else x
            return parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else "/static/default.jpg"
        except:
            return "/static/default.jpg"

    df["thumbnail"] = df["img_url"].apply(safe_extract_thumbnail)
    
    return render_template("house_list.html", houses=df.to_dict(orient="records"), query=query)


@app.route("/recommend_ui", methods=["GET"])
def recommend_ui():
    house_num_from_arg = request.args.get("house_num", type=int)
    top_k = request.args.get("top_k", default=10, type=int)

    if house_num_from_arg is None:
        return "<h2>house_num 파라미터가 필요합니다.🌈🦄</h2>", 400

    current_house_id = house_num_from_arg

    # Use a local DataFrame for this request to avoid issues with global df state
    # The connection 'conn' used here is the one defined with check_same_thread=False
    local_df = pd.read_sql("SELECT * FROM house", conn)

    row_match = local_df[local_df['house_num'] == current_house_id]
    if row_match.empty:
        return f"<h2>매물번호 {current_house_id} 에 해당하는 매물이 없습니다.</h2>", 404

    target_series = row_match.iloc[0]
    target_dict = target_series.to_dict() # Convert target to dict for template

    # Define extract_thumbnail helper function (as it was in the original scope)
    def extract_thumbnail(img_url_str):
        try:
            imgs = json.loads(img_url_str)
            return imgs[0] if imgs and isinstance(imgs, list) else None
        except (json.JSONDecodeError, TypeError):
            return None

    target_dict["thumbnail"] = extract_thumbnail(target_dict["img_url"])
    
    query_vec = np.array(json.loads(target_series['final_embedding']), dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query_vec)
    # Fetch k+1 to more easily remove the target item if it's in results, or handle if less are found
    D, I = index.search(query_vec, top_k + 5) # Fetch a bit more to allow for filtering

    # Process Faiss results robustly
    paired_results = []
    if D.size > 0 and I.size > 0:
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(local_df): # Ensure index is valid for local_df
                paired_results.append({'idx': idx, 'similarity': float(dist)})
    
    if not paired_results:
        result_df = pd.DataFrame(columns=list(local_df.columns) + ['similarity'])
    else:
        # Create DataFrame from valid Faiss results
        valid_indices = [res['idx'] for res in paired_results]
        valid_distances = [res['similarity'] for res in paired_results]
        
        result_df = local_df.iloc[valid_indices].copy()
        result_df['similarity'] = valid_distances

    # CRITICAL FIX POINT: Ensure 'house_num' in result_df is clean
    if not result_df.empty:
        # 1. Drop rows where 'house_num' is NaN (None)
        result_df.dropna(subset=['house_num'], inplace=True)
        
        # 2. Convert 'house_num' to numeric, coercing errors (non-numbers become NaN)
        result_df['house_num'] = pd.to_numeric(result_df['house_num'], errors='coerce')
        
        # 3. Drop rows where 'house_num' became NaN after coercion
        result_df.dropna(subset=['house_num'], inplace=True)
        
        # 4. Convert 'house_num' to int, if result_df is still not empty
        if not result_df.empty:
            result_df['house_num'] = result_df['house_num'].astype(int)
        else:
            # If all rows were dropped, ensure house_num column exists with int type for consistency
            result_df['house_num'] = pd.Series(dtype='int')

    # Filter out the query house itself, if present in results
    if not result_df.empty and 'house_num' in result_df.columns:
        result_df = result_df[result_df['house_num'] != current_house_id]
    
    # Apply other processing (duplicates, thumbnails)
    if not result_df.empty:
        result_df = result_df.drop_duplicates(subset=['title', 'address', 'gu', 'dong', 'price', 'area_size'])
        result_df["thumbnail"] = result_df["img_url"].apply(extract_thumbnail)
    
    # Limit to top_k results after all filtering
    result_df = result_df.head(top_k)

    # # items 출력을 위한 코드 추가
    # items = result_df.to_dict(orient="records")
    # print("\n=== 추천된 매물 목록 ===")
    # for idx, item in enumerate(items, 1):
    #     print(f"{idx}. {item['title']} | 가격: {item['price']} | 유사도: {item['similarity']}")

    return render_template(
        "recommend.html", 
        target=target_dict, 
        title=target_dict['title'], 
        house_num=current_house_id,
        items=result_df.to_dict(orient="records"),
        #items=items,
        loads=json.loads
    )

@app.template_filter('from_json')
def from_json_filter(s):
    try:
        return json.loads(s)
    except Exception:
        return []

if __name__ == "__main__":
    app.run(debug=True)
