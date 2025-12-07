import asyncio
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from events.Accident_detect import accident_detector as accident_det
from events.Accident_detect.incident_aggregator import AccidentAggregator
from events.bus import AsyncBus, topic_for
from events.frame_discrete import run_frame_source_raw, run_sampler_equal_time_vts

# ====== 路径配置 ======
BASE_DIR = r"E:\Training\traffic_video"

# 你的 fine-tuned 模型
FINETUNED_MODEL = (
    r"E:\PythonProject\DjangoTrafficAI\events\pts\best-4.pt"
)

# baseline：COCO 预训练 YOLOv8m（由 ultralytics 自动下载）
BASELINE_MODEL = "yolov8m.pt"


# 1. Parse labels from filename
def parse_labels_from_filename(path: str):
    """
    期望命名格式:
      accident-clear-cam-1.mp4
      noaccident-rain-cam-12.mp4
    只解析，不在当前评估中使用 weather。
    """
    name = os.path.basename(path).lower()
    if not name.endswith(".mp4"):
        raise ValueError(f"Not an mp4 file: {name}")
    name = name[:-4]

    parts = name.split("-")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename pattern: {name}")

    accident_str, weather, _, cam_id = parts
    has_acc = 1 if accident_str == "accident" else 0
    return f"cam-{cam_id}", has_acc, weather


# 2. Build CAMERA_SOURCES

def build_camera_sources():
    sources = {}
    for path in glob.glob(fr"{BASE_DIR}\*.mp4"):
        cam, gt_acc, weather = parse_labels_from_filename(path)
        sources[cam] = {
            "src": path,
            "gt": gt_acc,
            "weather": weather,
        }
    return sources



# 3. Evaluate single clip (full pipeline, with safe cleanup)

async def eval_single_clip(cam: str, src: str, gt: int):
    print(f"  - Evaluating {cam}")

    bus = AsyncBus()

    # Aggregator
    aggregator = AccidentAggregator(cam, bus, session_id=f"sess-{cam}")
    agg_task = asyncio.create_task(aggregator.run())

    # video-level predicted verdict
    predicted_accident = False

    async def event_listener():
        nonlocal predicted_accident
        topics = [
            topic_for("accidents.open", cam),
            topic_for("accidents.close", cam),  # 用于确认真正的生命周期
        ]
        async with bus.subscribe_many(topics, mode="fifo", maxsize=32) as sub:
            while True:
                evt = await sub.get()
                if evt.get("type") == "accident_open":
                    predicted_accident = True

    listener_task = asyncio.create_task(event_listener())

    # Pipeline
    frame_task = asyncio.create_task(run_frame_source_raw(bus, cam, src,simulate_realtime=False))
    sampler_task = asyncio.create_task(run_sampler_equal_time_vts(bus, cam, target_fps=15))
    detector_task = asyncio.create_task(
        accident_det.run_accident_detector_multi(bus, camera_ids=[cam], batch_size=4)
    )

    # 等待 frame_source 完全结束
    await frame_task

    # 给检测器 + 聚合器留出处理尾帧的时间
    await asyncio.sleep(1.0)

    # flush 聚合器：若事故未关闭会自动关闭
    await aggregator.flush()

    # -------- TASK CLEANUP ----------
    async def safe_cancel(task):
        if task:
            task.cancel()
            try:
                await task
            except:
                pass

    await safe_cancel(sampler_task)
    await safe_cancel(detector_task)
    await safe_cancel(agg_task)
    await safe_cancel(listener_task)

    print(f"    [RESULT] {cam}: GT={gt}, PRED={predicted_accident}")
    return 1 if predicted_accident else 0


# ----------------------------------------------------------
# 4. 评估某一个模型（finetuned / baseline）
# ----------------------------------------------------------
async def eval_model(model_name: str, model_path: str, cams, sources):
    """
    对所有视频用指定模型跑一遍，返回：
      y_true, y_pred, per_video_results, metrics_dict
    并在本地保存:
      confusion_matrix_{model_name}.png
      results_{model_name}.csv
    """
    print(f"\n=== Evaluating model: {model_name} ===")
    print(f"Using weights: {model_path}")

    # 替换 detector 里的模型路径
    accident_det._MODEL_PATH = model_path

    y_true, y_pred = [], []
    per_video = []

    for cam in cams:
        meta = sources[cam]
        gt = meta["gt"]
        pred = await eval_single_clip(cam, meta["src"], gt)
        y_true.append(gt)
        y_pred.append(pred)
        per_video.append(
            (cam, gt, pred, meta["weather"], meta["src"])
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))
    acc = (tp + tn) / len(y_true)

    print("\n  --- Metrics [{}] ---".format(model_name))
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Accuracy = {acc:.3f}")
    print(f"  Precision = {precision:.3f}")
    print(f"  Recall = {recall:.3f}")
    print(f"  F1 = {f1:.3f}")
    print("  ----------------------")

    # 混淆矩阵图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["GT 0", "GT 1"],
    )
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    fig_name = f"confusion_matrix_{model_name}.png"
    plt.savefig(fig_name, dpi=300)
    plt.close()
    print(f"  Saved {fig_name}")

    # 每视频结果表
    csv_name = f"results_{model_name}.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cam", "GT", "Pred", "Weather", "Path"])
        writer.writerows(per_video)
    print(f"  Saved {csv_name}")

    metrics = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    return y_true, y_pred, per_video, metrics


# ----------------------------------------------------------
# 5. 主流程：一次跑完 baseline + finetuned
# ----------------------------------------------------------
async def main():
    sources = build_camera_sources()
    cams = sorted(sources.keys())

    if not cams:
        print(f"No mp4 files found under {BASE_DIR}")
        return

    # 评估 finetuned 模型
    y_true_ft, y_pred_ft, per_video_ft, metrics_ft = await eval_model(
        "finetuned", FINETUNED_MODEL, cams, sources
    )

    # 评估 baseline 模型
    y_true_bs, y_pred_bs, per_video_bs, metrics_bs = await eval_model(
        "baseline", BASELINE_MODEL, cams, sources
    )

    # 基于 finetuned 结果导出 FP / FN 列表（方便你选典型错误案例）
    fp_fn_rows = []
    for (cam, gt, pred, weather, path) in per_video_ft:
        if gt == 1 and pred == 0:
            err_type = "FN"
        elif gt == 0 and pred == 1:
            err_type = "FP"
        else:
            continue
        fp_fn_rows.append([err_type, cam, gt, pred, weather, path])

    with open("fp_fn_cases_finetuned.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ErrType", "cam", "GT", "Pred", "Weather", "Path"])
        writer.writerows(fp_fn_rows)
    print("Saved fp_fn_cases_finetuned.csv")

    # baseline vs finetuned 对比表
    with open("baseline_vs_finetuned.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"]
        )
        writer.writerow(
            [
                "finetuned",
                metrics_ft["tp"],
                metrics_ft["fp"],
                metrics_ft["tn"],
                metrics_ft["fn"],
                f"{metrics_ft['accuracy']:.3f}",
                f"{metrics_ft['precision']:.3f}",
                f"{metrics_ft['recall']:.3f}",
                f"{metrics_ft['f1']:.3f}",
            ]
        )
        writer.writerow(
            [
                "baseline",
                metrics_bs["tp"],
                metrics_bs["fp"],
                metrics_bs["tn"],
                metrics_bs["fn"],
                f"{metrics_bs['accuracy']:.3f}",
                f"{metrics_bs['precision']:.3f}",
                f"{metrics_bs['recall']:.3f}",
                f"{metrics_bs['f1']:.3f}",
            ]
        )
    print("Saved baseline_vs_finetuned.csv")

    print("\n=== SUMMARY ===")
    print("Finetuned model:")
    print(
        f"  Acc={metrics_ft['accuracy']:.3f}, "
        f"Prec={metrics_ft['precision']:.3f}, "
        f"Rec={metrics_ft['recall']:.3f}, "
        f"F1={metrics_ft['f1']:.3f}"
    )
    print("Baseline model:")
    print(
        f"  Acc={metrics_bs['accuracy']:.3f}, "
        f"Prec={metrics_bs['precision']:.3f}, "
        f"Rec={metrics_bs['recall']:.3f}, "
        f"F1={metrics_bs['f1']:.3f}"
    )
    print("================")


if __name__ == "__main__":
    asyncio.run(main())
