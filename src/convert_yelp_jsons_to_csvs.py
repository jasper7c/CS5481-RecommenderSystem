import pandas as pd
import json
import os

def convert_json_to_csv(json_path, csv_path, max_lines=None):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
            if max_lines and i + 1 >= max_lines:
                break
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

def convert_all_yelp_jsons(base_dir, max_lines=100000):
    files = {
        "review": "yelp_academic_dataset_review.json",
        "business": "yelp_academic_dataset_business.json",
        "user": "yelp_academic_dataset_user.json"
    }

    for name, filename in files.items():
        json_path = os.path.join(base_dir, filename)
        csv_path = os.path.join(base_dir, f"{name}.csv")
        if os.path.exists(json_path):
            convert_json_to_csv(json_path, csv_path, max_lines=max_lines)
        else:
            print(f"File not found: {json_path}")

if __name__ == "__main__":
    # 替换成你自己的数据路径
    data_path = "../data/Yelp JSON/yelp_dataset"
    convert_all_yelp_jsons(data_path, max_lines=100000)  # 可调整为 None 加载全部
