# parse_bdd.py
import os, json
from .weather_map import map_weather

def load_bdd100k_weather(bdd_root, split="train"):
    img_dir  = os.path.join(bdd_root, "bdd100k_images_100k", "100k", split)
    lbl_dir  = os.path.join(bdd_root, "bdd100k_labels", "100k", split)

    samples = []

    for filename in os.listdir(img_dir):
        if not filename.endswith(".jpg"):
            continue

        stem = filename[:-4]  # 去掉 .jpg
        json_path = os.path.join(lbl_dir, stem + ".json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        raw_weather = data.get("attributes", {}).get("weather")
        mapped = map_weather(raw_weather)

        if mapped is None:
            continue

        samples.append((os.path.join(img_dir, filename), mapped))

    print(f"[BDD] {split}: {len(samples)} samples")
    return samples
