import os
import requests
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BlipProcessor,                      # 后面 BLIP-1 vs BLIP-2 对比用
    BlipForConditionalGeneration,
)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备:    {device}")

# float16 在 GPU 上大幅节省显存；CPU 上只能用 float32
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"精度:    {DTYPE}")

BLIP2_NAME  = "Salesforce/blip2-opt-2.7b"
BLIP2_CACHE = "./cache/models/blip2"
os.makedirs(BLIP2_CACHE, exist_ok=True)

# 检查本地缓存
slug = BLIP2_NAME.replace("/", "--")
cached = Path(BLIP2_CACHE).exists() and any(Path(BLIP2_CACHE).glob(f"models--{slug}*"))

if cached:
    print("[缓存命中] 从本地加载 BLIP-2")
else:
    print("[开始下载] BLIP-2 约 16 GB (float32) / 8 GB (float16 shards)")
    print("  如果下载卡住，检查代理后重新运行此 cell（支持断点续传）")

processor2 = Blip2Processor.from_pretrained(BLIP2_NAME, cache_dir=BLIP2_CACHE)

model2 = Blip2ForConditionalGeneration.from_pretrained(
    BLIP2_NAME,
    torch_dtype=DTYPE,      # float16 约 6GB 显存，float32 约 16GB
    cache_dir=BLIP2_CACHE,
).to(device)
model2.eval()

n_params = sum(p.numel() for p in model2.parameters()) / 1e9
print(f"\n模型加载完成！总参数量: {n_params:.2f}B")

# 展示三个组件的参数量分布
components = {
    "vision_model (冻结)": model2.vision_model,
    "qformer (可训练)": model2.qformer,
    "language_model (冻结)": model2.language_model,
}
print("\n各组件参数量：")
for name, module in components.items():
    n = sum(p.numel() for p in module.parameters()) / 1e6
    print(f"  {name:30s}: {n:.0f}M")
    
IMAGE_DIR = "./cache/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def load_image(url: str, filename: str) -> Image.Image:
    """下载并缓存图片，已存在则直接读取"""
    path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(path):
        print(f"[缓存] {filename}")
    else:
        print(f"[下载] {filename} ...")
        try:
            r = requests.get(url, timeout=30, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"下载失败: {e}\n请检查代理或手动下载图片到 {path}")
    return Image.open(path).convert("RGB")


images = {
    "海滩女孩与狗": load_image(
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
        "demo.jpg",
    ),
    "胖猫": load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "cat.jpg",
    ),
    "城市夜景": load_image(
        "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800",
        "city_night.jpg",
    ),
    "咖啡拉花": load_image(
        "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=800",
        "coffee.jpg",
    ),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, img) in zip(axes, images.items()):
    ax.imshow(img)
    ax.set_title(name, fontsize=10)
    ax.axis("off")
plt.suptitle("测试图片", fontsize=12)
plt.tight_layout()
plt.show()


def caption(model, proc, image, device, dtype, max_new_tokens=50, num_beams=5):
    """无条件图像描述生成"""
    inputs = proc(images=image, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


print("=== BLIP-2 图像描述 ===")
for name, img in images.items():
    cap = caption(model2, processor2, img, device, DTYPE)
    print(f"\n【{name}】\n  {cap}")
    
    
def vqa(model, proc, image, question, device, dtype, max_new_tokens=30):
    """视觉问答。返回 (完整输出, 仅答案部分)"""
    prompt = f"Question: {question} Answer:"
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    full  = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    # 截取 "Answer:" 之后的部分
    answer = full.split("Answer:")[-1].strip()
    return answer


img = images["海滩女孩与狗"]
questions = [
    "What is the woman doing?",
    "What animal is with the woman?",
    "What time of day does it appear to be?",
    "Describe the mood of this scene.",
    "If you had to write a news headline for this photo, what would it be?",
]

print("=== 对「海滩女孩与狗」提问 ===\n")
for q in questions:
    ans = vqa(model2, processor2, img, q, device, DTYPE)
    print(f"Q: {q}")
    print(f"A: {ans}\n")
    
    
# 再试试其他图片
img_coffee = images["咖啡拉花"]
questions_coffee = [
    "What drink is this?",
    "What art is on top of the drink?",
    "What kind of shop would serve this?",
    "Can you estimate the price of this drink?",
]

print("=== 对「咖啡拉花」提问 ===\n")
for q in questions_coffee:
    ans = vqa(model2, processor2, img_coffee, q, device, DTYPE)
    print(f"Q: {q}")
    print(f"A: {ans}\n")