import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import json

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    scaler = GradScaler()
    model.train()
    # 内部初始化scheduler（使用传入的optimizer）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    train_metrics = {"loss": [], "miou": []}
    val_metrics = {"loss": [], "miou": [], "dice": [], "pa": []}
    
    for epoch in range(epochs):
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=total_train_loss/len(train_loader))
        
        train_metrics["loss"].append(total_train_loss / len(train_loader))
        
        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        total_metrics = {"miou": 0.0, "dice": 0.0, "pa": 0.0}
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                metrics = calculate_metrics(preds, masks, num_classes=28)
                total_metrics["miou"] += metrics["miou"]
                total_metrics["dice"] += metrics["dice"]
                total_metrics["pa"] += metrics["pa"]
        
        val_metrics["loss"].append(total_val_loss / len(test_loader))
        val_metrics["miou"].append(total_metrics["miou"] / len(test_loader))
        val_metrics["dice"].append(total_metrics["dice"] / len(test_loader))
        val_metrics["pa"].append(total_metrics["pa"] / len(test_loader))
        
        scheduler.step(total_val_loss)  # 使用内部scheduler
        model.train()
    
    return train_metrics, val_metrics

def evaluate(model, test_loader, device, num_classes, return_confusion=False):
    """修改：返回完整指标（mIoU、Dice、PA）"""
    model.eval()
    total_metrics = {"miou": 0.0, "dice": 0.0, "pa": 0.0}
    conf_matrix = torch.zeros(num_classes, num_classes, device=device) if return_confusion else None

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # 计算指标
            metrics = calculate_metrics(preds, masks, num_classes)
            total_metrics["miou"] += metrics["miou"]
            total_metrics["dice"] += metrics["dice"]
            total_metrics["pa"] += metrics["pa"]
            if return_confusion:
                for cls in range(num_classes):
                    for pred_cls in range(num_classes):
                        conf_matrix[cls, pred_cls] += ((masks == cls) & (preds == pred_cls)).sum()

    # 平均指标
    avg_metrics = {k: v / len(test_loader) for k, v in total_metrics.items()}
    return avg_metrics, conf_matrix.cpu().numpy() if return_confusion else avg_metrics

def visualize_results(image, mask, pred_mask, class_map=None, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    print(f"Image shape before processing: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Pred mask shape: {pred_mask.shape}")
    
    colors = list(mcolors.TABLEAU_COLORS.values())[:28]
    cmap = mcolors.ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    image_vis = image.squeeze(0).permute(1, 2, 0).numpy() 
    axes[0].imshow(image_vis)
    axes[0].set_title("Input Image")
    
    # 真实掩码
    axes[1].imshow(mask, cmap=cmap)
    axes[1].set_title("Ground Truth Mask")
    
    # 预测掩码
    axes[2].imshow(pred_mask, cmap=cmap)
    axes[2].set_title("Predicted Mask")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def save_train_curve(train_metrics, val_metrics, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics["loss"], label="Train Loss")
    plt.plot(val_metrics["loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(conf_matrix, class_map, model_name, output_dir):
    plt.figure(figsize=(15, 15))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks(range(len(class_map)), class_map.values(), rotation=45)
    plt.yticks(range(len(class_map)), class_map.values())
    plt.colorbar()
    save_path = os.path.join(output_dir, f"{model_name}_confusion.png")
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics_path, train_metrics, val_metrics):
    """修改：支持保存新增的miou、dice、pa指标"""
    metrics_data = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
def calculate_metrics(preds, masks, num_classes):
    """计算mIoU、Dice系数、像素精度（PA）"""
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)
    total_correct = 0
    total_pixels = 0

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (masks == cls)
        
        # 计算交集和并集（用于mIoU和Dice）
        intersection[cls] = (pred_cls & target_cls).sum()
        union[cls] = (pred_cls | target_cls).sum()
        
        # 计算像素精度（PA）
        total_correct += (pred_cls == target_cls).sum()
        total_pixels += target_cls.numel()

    # 计算mIoU（忽略分母为0的类别）
    iou = intersection / (union + 1e-6)
    miou = iou[~torch.isnan(iou)].mean()

    # 计算Dice系数（全局平均）
    dice = (2 * intersection) / (intersection + union + 1e-6)
    avg_dice = dice[~torch.isnan(dice)].mean()

    # 计算像素精度（PA）
    pa = total_correct / total_pixels

    return {
        "miou": miou.item(),
        "dice": avg_dice.item(),
        "pa": pa.item()
    }