"""CLIP training script with real CIFAR-10 data"""
import os

# 先清除所有代理变量，然后设置HTTP代理
for v in ['all_proxy', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(v, None)
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


class Config:
    MODEL_NAME = "openai/clip-vit-base-patch32"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    SAVE_DIR = "./outputs/clip"


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = torch.matmul(image_features, text_features.T) / self.temperature
        logits_per_text = logits_per_image.T
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)
        loss_i = self.cross_entropy_loss(logits_per_image, labels)
        loss_t = self.cross_entropy_loss(logits_per_text, labels)
        return (loss_i + loss_t) / 2


def get_features(model, image_inputs=None, text_inputs=None):
    image_features = None
    text_features = None
    if image_inputs is not None:
        img_out = model.get_image_features(**image_inputs)
        image_features = img_out.pooler_output if hasattr(img_out, 'pooler_output') else img_out
    if text_inputs is not None:
        text_out = model.get_text_features(**text_inputs)
        text_features = text_out.pooler_output if hasattr(text_out, 'pooler_output') else text_out
    return image_features, text_features


def train_step(model, processor, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            images = batch["image"].to(device)
            texts = batch["caption"]
            
            text_inputs = processor(text=texts, padding=True, return_tensors="pt").to(device)
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            
            with torch.cuda.amp.autocast():
                image_features, text_features = get_features(model, image_inputs, text_inputs)
                loss_fn = ContrastiveLoss()
                loss = loss_fn(image_features, text_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": loss.item()})
        except Exception as e:
            print(f"\n批次 {batch_idx} 出错: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0


def main():
    print("=" * 60)
    print("CLIP 模型训练")
    print("=" * 60)
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"模型: {Config.MODEL_NAME}")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print("=" * 60)
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    print(f"\n正在加载模型...")
    model = CLIPModel.from_pretrained(Config.MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(Config.MODEL_NAME)
    print("模型加载完成!")
    
    # 使用 CIFAR-10 数据集
    print(f"\n正在加载 CIFAR-10 数据集...")
    from datasets import load_dataset
    
    dataset = load_dataset("cifar10", split="train")
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                     'dog', 'frog', 'horse', 'ship', 'truck']
    
    def add_caption(example, idx):
        example['caption'] = f"a photo of a {cifar10_labels[example['label']]}"
        example['image'] = example['img']
        return example
    
    dataset = dataset.map(add_caption, with_indices=True)
    dataset = dataset.select(range(800))
    
    print(f"数据集大小: {len(dataset)}")
    
    # 自定义 collate 函数
    def collate_fn(batch):
        images = torch.stack([torch.from_numpy(np.array(item["image"])).permute(2, 0, 1).float() for item in batch])
        captions = [item["caption"] for item in batch]
        return {"image": images, "caption": captions}
    
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"\n开始训练...")
    
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_step(model, processor, train_loader, optimizer, device, epoch)
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        
        save_path = os.path.join(Config.SAVE_DIR, f"clip_epoch_{epoch + 1}.pt")
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        print(f"  模型已保存: {save_path}")
    
    print(f"\n训练完成!")
    print(f"模型保存在: {Config.SAVE_DIR}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA 支持"
    main()
