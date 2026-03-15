from pathlib import Path

# ====== 修改成你的数据集根目录 ======
DATASET_ROOT = Path(r"E:\Training\Acci_Dataset")
# 例如:
# DATASET_ROOT = Path(r"D:\datasets\accident_dataset")
# 目录结构应类似:
# train/images, train/labels
# valid/images, valid/labels
# test/images, test/labels

SPLITS = ["train", "valid", "test"]

# 原始类别定义：按你当前数据集来
ACCIDENT_CLASS_IDS = {0}
NON_ACCIDENT_CLASS_IDS = {1}

def process_label_file(label_path: Path):
    """
    删除 non-accident 标注，只保留 accident。
    并把保留的类别统一重映射为 0。
    """
    original_lines = label_path.read_text(encoding="utf-8").splitlines()

    kept_lines = []
    removed_count = 0
    kept_count = 0
    unknown_count = 0

    for raw_line in original_lines:
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            # 非法行，直接跳过
            unknown_count += 1
            continue

        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            unknown_count += 1
            continue

        coords = parts[1:]

        if cls_id in NON_ACCIDENT_CLASS_IDS:
            removed_count += 1
            continue

        if cls_id in ACCIDENT_CLASS_IDS:
            # 统一重映射为 0
            kept_lines.append("0 " + " ".join(coords))
            kept_count += 1
        else:
            # 未知类别，默认丢弃；你也可以改成保留
            unknown_count += 1

    # 写回文件：保留为空文件，表示负样本无框
    new_content = "\n".join(kept_lines)
    if new_content:
        new_content += "\n"
    label_path.write_text(new_content, encoding="utf-8")

    return {
        "file": str(label_path),
        "original_lines": len(original_lines),
        "kept": kept_count,
        "removed_non_accident": removed_count,
        "unknown_or_invalid": unknown_count,
        "empty_after": len(kept_lines) == 0,
    }


def main():
    total_files = 0
    total_kept = 0
    total_removed = 0
    total_unknown = 0
    total_empty_after = 0

    for split in SPLITS:
        labels_dir = DATASET_ROOT / split / "labels"
        if not labels_dir.exists():
            print(f"[跳过] 不存在目录: {labels_dir}")
            continue

        txt_files = sorted(labels_dir.glob("*.txt"))
        print(f"\n=== 处理 {split} / labels ===")
        print(f"找到 {len(txt_files)} 个标签文件")

        for label_file in txt_files:
            result = process_label_file(label_file)
            total_files += 1
            total_kept += result["kept"]
            total_removed += result["removed_non_accident"]
            total_unknown += result["unknown_or_invalid"]
            total_empty_after += int(result["empty_after"])

        print(f"{split} 处理完成")

    print("\n===== 汇总 =====")
    print(f"处理标签文件数: {total_files}")
    print(f"保留 accident 标注数: {total_kept}")
    print(f"删除 non-accident 标注数: {total_removed}")
    print(f"未知/非法标注数: {total_unknown}")
    print(f"处理后为空的标签文件数: {total_empty_after}")
    print("\n完成。non-accident 标签已移除，图片仍可作为无框负样本使用。")


if __name__ == "__main__":
    main()