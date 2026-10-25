"""
plot_destructive_swap.py - quick self-check plot: error versus shots, from results/swap_summary.csv.
Saves results/destructive_swap_error_vs_shots.png and results/swap_error_vs_shots.png.
"""
import csv
import math
import os
import numpy as np


def load_summary(path="results/destructive_swap_summary.csv"):
    if not os.path.exists(path):
        alt = "results/swap_summary.csv"
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(f"Neither {path} nor {alt} found. Run experiment first.")
    shots = []
    rmse = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shots.append(float(row["shots"]))
            rmse.append(float(row["rmse"]))
    return np.array(shots), np.array(rmse)


def plot_with_matplotlib(shots, rmse, out_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(shots, rmse, "o-", color="#1f77b4", label="destructive SWAP (RMSE)")
    ref = rmse[0] * np.sqrt(shots[0] / shots)
    ax.loglog(shots, ref, "--", color="gray", label="1/sqrt(shots) reference")
    ax.set_xlabel("total shots")
    ax.set_ylabel("RMSE of overlap estimate")
    ax.set_title("Destructive SWAP test, ideal simulation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    for p in out_paths:
        plt.savefig(p, dpi=150)
        print(f"saved {p}")
    plt.close()


def plot_with_pil(shots, rmse, out_paths):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 900, 600
    margin_l, margin_r = 110, 50
    margin_t, margin_b = 60, 80

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    ref = rmse[0] * np.sqrt(shots[0] / shots)

    min_x, max_x = np.min(shots), np.max(shots)
    all_y = np.concatenate([rmse, ref])
    min_y, max_y = np.min(all_y) * 0.7, np.max(all_y) * 1.3

    log_min_x, log_max_x = np.log10(min_x), np.log10(max_x)
    log_min_y, log_max_y = np.log10(min_y), np.log10(max_y)

    def to_pixel(x, y):
        px = margin_l + (np.log10(x) - log_min_x) / (log_max_x - log_min_x) * plot_w
        py = margin_t + plot_h - (np.log10(y) - log_min_y) / (log_max_y - log_min_y) * plot_h
        return px, py

    # Draw grid and ticks
    for exp in range(int(np.floor(log_min_x)), int(np.ceil(log_max_x)) + 1):
        v = 10 ** exp
        if min_x <= v <= max_x:
            px, _ = to_pixel(v, min_y)
            draw.line([(px, margin_t), (px, margin_t + plot_h)], fill="#e0e0e0", width=1)
            draw.text((px - 15, margin_t + plot_h + 10), f"10^{exp}", fill="#333333")

    for exp in range(int(np.floor(log_min_y)), int(np.ceil(log_max_y)) + 1):
        v = 10 ** exp
        if min_y <= v <= max_y:
            _, py = to_pixel(min_x, v)
            draw.line([(margin_l, py), (margin_l + plot_w, py)], fill="#e0e0e0", width=1)
            draw.text((margin_l - 60, py - 6), f"10^{exp}", fill="#333333")

    # Draw axes box
    draw.rectangle(
        [(margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h)],
        outline="#333333", width=2
    )

    # Reference line (1/sqrt(shots))
    ref_points = [to_pixel(x, y) for x, y in zip(shots, ref)]
    for i in range(len(ref_points) - 1):
        draw.line([ref_points[i], ref_points[i+1]], fill="#888888", width=2)

    # Data points and connecting line
    data_points = [to_pixel(x, y) for x, y in zip(shots, rmse)]
    for i in range(len(data_points) - 1):
        draw.line([data_points[i], data_points[i+1]], fill="#1f77b4", width=3)

    r = 5
    for px, py in data_points:
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill="#1f77b4", outline="#0d47a1")

    # Title & labels
    draw.text((width // 2 - 140, 20), "Destructive SWAP test, ideal simulation", fill="#111111")
    draw.text((width // 2 - 35, height - 35), "total shots", fill="#111111")
    draw.text((15, height // 2 - 20), "RMSE", fill="#111111")

    # Legend
    leg_x, leg_y = margin_l + plot_w - 260, margin_t + 25
    draw.rectangle([(leg_x, leg_y), (leg_x + 245, leg_y + 60)], fill="#ffffff", outline="#cccccc")
    draw.line([(leg_x + 10, leg_y + 18), (leg_x + 35, leg_y + 18)], fill="#1f77b4", width=3)
    draw.ellipse([(leg_x + 20, leg_y + 15), (leg_x + 26, leg_y + 21)], fill="#1f77b4")
    draw.text((leg_x + 45, leg_y + 12), "destructive SWAP (RMSE)", fill="#222222")

    draw.line([(leg_x + 10, leg_y + 40), (leg_x + 35, leg_y + 40)], fill="#888888", width=2)
    draw.text((leg_x + 45, leg_y + 34), "1/sqrt(shots) reference", fill="#222222")

    for p in out_paths:
        img.save(p)
        print(f"saved {p}")


def main():
    shots, rmse = load_summary()
    out_paths = ["results/destructive_swap_error_vs_shots.png"]
    try:
        plot_with_matplotlib(shots, rmse, out_paths)
    except Exception as e:
        # Graceful fallback to PIL when C-extension DLL is blocked by OS policy
        plot_with_pil(shots, rmse, out_paths)


if __name__ == "__main__":
    main()
