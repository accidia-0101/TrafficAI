"""
TrafficAI Performance Monitor (Standalone Simple Version)
--------------------------------------------------------
- 独立进程运行，不依赖 BUS，不影响你的 Django / YOLO 管线。
- 按时间轴持续记录到 CSV：
    * CPU usage (%)
    * System memory usage (%)
    * GPU usage (% via nvidia-smi)
    * GPU memory used (MB via nvidia-smi)

Usage:
    python performance_monitor.py

Note:
    - 需要系统有 NVIDIA 驱动，并且能在命令行执行 `nvidia-smi`
    - 不再依赖 NVML / pynvml，解决 NVMLError_LibraryNotFound 问题
"""

import time
import csv
import psutil
import subprocess

# ==========================
# 配置
# ==========================
LOG_FILE = "performance_log.csv"   # 输出 CSV 文件名
LOG_INTERVAL = 0.5                 # 记录间隔（秒）
# ==========================


# ==========================
# GPU 监控（通过 nvidia-smi）
# ==========================
def get_gpu_stats():
    """
    返回 (gpu_util_percent, gpu_mem_used_MB)
    如果系统没有 NVIDIA GPU 或 nvidia-smi 不可用，返回 (0.0, 0.0)
    """
    try:
        # 一次性查询利用率和显存使用，避免两次调用
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        # 可能有多块卡，这里只取第一行第一块
        first_line = out.strip().splitlines()[0]
        util_str, mem_str = [x.strip() for x in first_line.split(",")]
        gpu_util = float(util_str)        # %
        gpu_mem = float(mem_str)         # MB
        return gpu_util, gpu_mem
    except Exception:
        # 没有 NVIDIA / 没有 nvidia-smi / 其他错误 → 返回 0
        return 0.0, 0.0


# ==========================
# CPU / 内存
# ==========================
def get_cpu_usage():
    # 整体 CPU 使用率（所有核心平均）
    return psutil.cpu_percent()


def get_memory_usage():
    mem = psutil.virtual_memory()
    return float(mem.percent)


# ==========================
# 主循环：持续写 CSV
# ==========================
def main():
    print(f"[MONITOR] Logging to {LOG_FILE!r} (interval={LOG_INTERVAL}s)")

    # 创建 CSV 文件 & header
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",       # Unix 时间戳（秒）
                "cpu_percent",     # CPU 使用率（整体，%）
                "memory_percent",  # 系统内存使用率（%）
                "gpu_percent",     # GPU 使用率（%）
                "gpu_memory_MB",   # GPU 已用显存（MB）
            ]
        )

    try:
        while True:
            ts = time.time()

            cpu = get_cpu_usage()
            mem = get_memory_usage()
            gpu, gpu_mem = get_gpu_stats()

            # 追加写入一行
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ts, cpu, mem, gpu, gpu_mem])

            # 也可以在控制台简单打印一行（不想看可以删掉这几行）
            print(
                f"[MONITOR] t={ts:.0f}  CPU={cpu:5.1f}%  MEM={mem:5.1f}%  "
                f"GPU={gpu:5.1f}%  GPU-MEM={gpu_mem:7.1f}MB",
                end="\r",
            )

            time.sleep(LOG_INTERVAL)

    except KeyboardInterrupt:
        print("\n[MONITOR] Stopped by user.")


if __name__ == "__main__":
    main()
