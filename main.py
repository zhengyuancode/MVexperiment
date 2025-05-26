import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import D4LADataset
from model import UNet, UNet_Attention 
from train_eval import (
    train_model,
    evaluate,
    visualize_results,
    plot_confusion_matrix
)
from utils import ensure_dir, generate_metrics_table

# 实验配置
output_root = "experiment_results"
ensure_dir(output_root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
image_size = 256
num_classes = 28  # 0-27（背景+27类）
epochs = 50

# 类别映射
with open("data/D4LA/json/map_info.json", "r") as f:
    raw_class_map = json.load(f)["categories"]
class_map = {0: "Background"}
class_map.update({v: k for k, v in raw_class_map.items()})

# 数据变换
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# 数据集加载
train_dataset = D4LADataset(
    image_dir="data/D4LA/train_images",
    json_path="data/D4LA/json/train.json",
    class_map=class_map,
    transform=transform
)
test_dataset = D4LADataset(
    image_dir="data/D4LA/test_images",
    json_path="data/D4LA/json/test.json",
    class_map=class_map,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 模型定义与训练
def run_experiment(model_name, model):
    model = model.to(device)
    
    visualize_before(model, test_dataset, model_name, output_root)
    
    train_metrics, val_metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        device=device,
        epochs=epochs
    )
    
    visualize_after(model, test_dataset, model_name, output_root)
    
    return train_metrics, val_metrics

def visualize_before(model, dataset, model_name, output_dir):
    image, mask = dataset[0]
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image)
    pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    save_path = os.path.join(output_dir, f"{model_name}_before.png")
    visualize_results(image.cpu(), mask, pred_mask, class_map, save_path)

def visualize_after(model, dataset, model_name, output_dir):
    image, mask = dataset[0]
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image)
    pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    save_path = os.path.join(output_dir, f"{model_name}_after.png")
    visualize_results(image.cpu(), mask, pred_mask, class_map, save_path)

def print_tables(baseline_train_metrics, baseline_val_metrics, attention_train_metrics, attention_val_metrics):
    # 表格 1 数据（训练轮次的 mIoU）
    table1_data = {
        "训练轮次": [10, 20, 30, 40, 50],
        "U-Net（mIoU）": [baseline_val_metrics["miou"][i-1] for i in [10, 20, 30, 40, 50]],
        "Attention U-Net（mIoU）": [attention_val_metrics["miou"][i-1] for i in [10, 20, 30, 40, 50]],
        "提升幅度": []
    }

    for i in range(len(table1_data["训练轮次"])):
        improvement = ((table1_data["Attention U-Net（mIoU）"][i] - table1_data["U-Net（mIoU）"][i]) / table1_data["U-Net（mIoU）"][i]) * 100
        table1_data["提升幅度"].append(f"+{improvement:.1f}%")

    # 表格 2 数据（最终评估指标）
    table2_data = {
        "评估指标": ["mIoU", "Dice 系数", "像素精度（PA）"],
        "U-Net": [baseline_val_metrics["miou"][-1], baseline_val_metrics["dice"][-1], baseline_val_metrics["pa"][-1]],
        "Attention U-Net": [attention_val_metrics["miou"][-1], attention_val_metrics["dice"][-1], attention_val_metrics["pa"][-1]],
        "提升幅度": []
    }

    for i in range(len(table2_data["评估指标"])):
        improvement = ((table2_data["Attention U-Net"][i] - table2_data["U-Net"][i]) / table2_data["U-Net"][i]) * 100
        table2_data["提升幅度"].append(f"+{improvement:.1f}%")

    print("\n表格 1：")
    print(f"{'训练轮次':<10} {'U-Net（mIoU）':<15} {'Attention U-Net（mIoU）':<25} {'提升幅度':<10}")
    for i in range(len(table1_data["训练轮次"])):
        print(f"{table1_data['训练轮次'][i]:<10} {table1_data['U-Net（mIoU）'][i]:<15.3f} {table1_data['Attention U-Net（mIoU）'][i]:<25.3f} {table1_data['提升幅度'][i]:<10}")

    print("\n表格 2：")
    print(f"{'评估指标':<15} {'U-Net':<15} {'Attention U-Net':<20} {'提升幅度':<10}")
    for i in range(len(table2_data["评估指标"])):
        print(f"{table2_data['评估指标'][i]:<15} {table2_data['U-Net'][i]:<15.3f} {table2_data['Attention U-Net'][i]:<20.3f} {table2_data['提升幅度'][i]:<10}")

if __name__ == "__main__":
    # 运行基线模型（无注意力）
    baseline_train_metrics, baseline_val_metrics = run_experiment("baseline", UNet(n_classes=num_classes))
    
    # 运行带注意力模型
    attention_train_metrics, attention_val_metrics = run_experiment("attention", UNet_Attention(n_classes=num_classes))
    
    print_tables(baseline_train_metrics, baseline_val_metrics, attention_train_metrics, attention_val_metrics)