import os
import json
import torch
from train_eval import evaluate
from tabulate import tabulate

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def generate_metrics_table(output_root):
    # 读取基线模型和注意力模型的指标
    with open(os.path.join(output_root, "baseline_metrics.json"), "r") as f:
        baseline_metrics = json.load(f)["val_metrics"] 
    
    with open(os.path.join(output_root, "attention_metrics.json"), "r") as f:
        attention_metrics = json.load(f)["val_metrics"]
    
    with open(os.path.join(output_root, "baseline_final_metrics.json"), "r") as f:
        baseline_final = json.load(f)
    
    with open(os.path.join(output_root, "attention_final_metrics.json"), "r") as f:
        attention_final = json.load(f)

    # 生成训练轮次对比表
    rounds = [10, 20, 30, 40, 50]
    table_data = []
    for round in rounds:
        if round > len(baseline_metrics["miou"]):
            continue 
        baseline_miou = baseline_metrics["miou"][round-1]
        attention_miou = attention_metrics["miou"][round-1]
        improvement = ((attention_miou - baseline_miou) / baseline_miou) * 100
        table_data.append([
            round,
            f"{baseline_miou:.2f}",
            f"{attention_miou:.2f}",
            f"+{improvement:.1f}%"
        ])
    
    # 生成评估指标对比表
    metric_table = [
        ["mIoU", f"{baseline_final['miou']:.2f}", f"{attention_final['miou']:.2f}", f"+{((attention_final['miou']-baseline_final['miou'])/baseline_final['miou'])*100:.1f}%"],
        ["Dice 系数", f"{baseline_final['dice']:.2f}", f"{attention_final['dice']:.2f}", f"+{((attention_final['dice']-baseline_final['dice'])/baseline_final['dice'])*100:.1f}%"],
        ["像素精度（PA）", f"{baseline_final['pa']:.2f}", f"{attention_final['pa']:.2f}", f"+{((attention_final['pa']-baseline_final['pa'])/baseline_final['pa'])*100:.1f}%"]
    ]

    # 输出到文件
    with open(os.path.join(output_root, "metrics_table.txt"), "w") as f:
        # 训练轮次对比表
        f.write("训练轮次对比表：\n")
        f.write(tabulate(table_data, headers=["训练轮次", "U-Net（mIoU）", "Attention U-Net（mIoU）", "提升幅度"], tablefmt="grid"))
        f.write("\n\n")
        
        # 评估指标对比表
        f.write("评估指标对比表：\n")
        f.write(tabulate(metric_table, headers=["评估指标", "U-Net", "Attention U-Net", "提升幅度"], tablefmt="grid"))

def save_metrics(path, train_metrics, val_metrics):
    with open(path, "w") as f:
        json.dump({
            "train": train_metrics,
            "val": val_metrics
        }, f)

def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)