import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class D4LADataset(Dataset):
    def __init__(self, image_dir, json_path, class_map, transform=None, mode="train"):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.class_map = class_map 
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 构建图像信息字典
        self.images = {img["id"]: img for img in data["images"]}
        # 构建标注字典
        self.annotations = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # 有效图像id就是同时存在于images和annotations中的id
        self.valid_ids = [img_id for img_id in self.images if img_id in self.annotations]
        print(f"初始化D4LADataset，有效图像数量: {len(self.valid_ids)}")  

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size  
        # 初始化掩码为0（背景类），尺寸与原图一致
        mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        
        for ann in self.annotations[img_id]:
            category_id = ann["category_id"]
            if not (1 <= category_id <= len(self.class_map)):
                print(f"警告：图像{img_info['file_name']}的标注类别ID{category_id}无效，跳过！")
                continue
            
            # 转换并闭合多边形
            poly = np.array(ann["poly"], dtype=np.int32).reshape((-1, 2))
            if not np.all(poly[0] == poly[-1]):
                poly = np.vstack([poly, poly[0]]) 
            
            # 限制坐标在图像范围内
            poly[:,0] = np.clip(poly[:,0], 0, orig_width)
            poly[:,1] = np.clip(poly[:,1], 0, orig_height)
            
            try:
                cv2.fillPoly(mask, [poly], color=category_id)
            except Exception as e:
                print(f"填充失败！图像{img_info['file_name']}的标注{ann['id']}错误: {e}")
        
        # 转换为PIL图像并应用变换
        mask_pil = Image.fromarray(mask)
        if self.transform:
            image = self.transform(image) 
            mask_pil = transforms.Resize((256, 256), interpolation=Image.NEAREST)(mask_pil)
            # 直接转换为整数张量（不除以255）
            mask_tensor = transforms.PILToTensor()(mask_pil).squeeze(0).long()
        else:
            mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        print(f"图像{img_info['file_name']}变换后的掩码唯一值: {torch.unique(mask_tensor)}")
        return image, mask_tensor