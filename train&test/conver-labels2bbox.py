import os

# 你的数据集根目录，例如：
ROOT = r"E:\Training\CCD-DATA"

SUBSETS = ["train", "valid", "test"]

def process_label_file(path):
    # 如果标签为空（无事故） -> 保持不动
    if os.path.getsize(path) == 0:
        return

    # 有事故 -> 统一替换成全图 bbox
    with open(path, "w") as f:
        f.write("0 0.5 0.5 1.0 1.0\n")


def convert_dataset(root):
    for subset in SUBSETS:
        label_dir = os.path.join(root, subset, "labels")
        if not os.path.exists(label_dir):
            print(f"跳过不存在的目录: {label_dir}")
            continue

        print(f"处理目录: {label_dir}")

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            label_path = os.path.join(label_dir, fname)
            process_label_file(label_path)

        print(f"完成: {label_dir}")


if __name__ == "__main__":
    convert_dataset(ROOT)
    print("所有标签文件已处理完毕！")
