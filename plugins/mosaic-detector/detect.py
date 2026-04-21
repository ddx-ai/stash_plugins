import requests
import cv2
import numpy as np
import os

STASH_URL = "http://127.0.0.1:9999/graphql" 
API_KEY = "" 
TARGET_TAG = "Mosaic" 

def stash_query(query, variables=None):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["ApiKey"] = API_KEY
    payload = {'query': query}
    if variables:
        payload['variables'] = variables
    try:
        res = requests.post(STASH_URL, json=payload, headers=headers)
        # 400エラーが起きた際、Stashが返している具体的なエラー文を出力する
        if res.status_code == 400:
            print(f"!!! GraphQL Syntax Error (400) !!!")
            print(f"Request Query: {query[:100]}...")
            print(f"Response Text: {res.text}")
            return None
        return res.json().get('data')
    except Exception as e:
        print(f"Connection failed: {e}")
        return None

def main():
    print("--- Starting Mosaic Detector Debug Mode ---")
    
    # 1. タグの存在確認
    data = stash_query('{ allTags { id name } }')
    if data is None:
        print("Failed to fetch tags.")
        return
    
    tags = data.get('allTags', [])
    tag_id = next((t['id'] for t in tags if t['name'] == TARGET_TAG), None)

    # 2. タグ作成 (ここが400の可能性大)
    if not tag_id:
        print(f"Tag '{TARGET_TAG}' not found. Attempting to create...")
        # 構文を極限まで削ぎ落としたMutation
        create_query = 'mutation TagCreate($name: String!) { tagCreate(input: { name: $name }) { id } }'
        c_data = stash_query(create_query, {"name": TARGET_TAG})
        if c_data:
            tag_id = c_data['tagCreate']['id']
            print(f"Success! Tag Created ID: {tag_id}")
        else:
            print("Failed at Tag Creation.")
            return

    # 3. 画像取得 (ここが400の可能性もあり)
    print("Fetching images...")
    # フィールドを1つだけに絞ってテスト
    i_data = stash_query('{ allImages { id } }')
    if i_data is None:
        print("Failed at Image Fetching.")
        return

    print(f"Success! Found {len(i_data.get('allImages', []))} images.")
    print("--- Debug End ---")

if __name__ == "__main__":
    main()
