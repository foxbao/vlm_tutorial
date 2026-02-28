import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["DISABLE_TELEMETRY"] = "YES"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests




# 使用本地缓存，不检查网络
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DISABLE_AUTO_CONVERSION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 禁用 transformers 的自动转换后台线程
os.environ["TRANSFORMERS_DISABLE_AUTO_CONVERSION"] = "1"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("All imports successful!")

# 加载 CLIP 模型和处理器
print("正在加载 CLIP 模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
print("CLIP 模型加载完成！")

# 下载一张测试图片
print("\n正在下载测试图片...")
url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
print("图片下载完成！")

# 图像编码
print("\n正在编码图像...")
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
# get_image_features 返回 BaseModelOutputWithPooling 对象
# pooler_output 是经过池化的特征向量 (batch_size, hidden_size)
print(f"图像特征类型: {type(image_features)}")
print(f"池化特征形状 (pooler_output): {image_features.pooler_output.shape}")
print(f"最后隐藏状态形状 (last_hidden_state): {image_features.last_hidden_state.shape}")
print("图像编码完成！")

# 文本编码
print("\n正在编码文本...")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
text_inputs = processor(text=texts, padding=True, return_tensors="pt")
text_features = model.get_text_features(**text_inputs)
# get_text_features 也返回 BaseModelOutputWithPooling 对象
print(f"文本特征类型: {type(text_features)}")
print(f"池化特征形状 (pooler_output): {text_features.pooler_output.shape}")
print(f"最后隐藏状态形状 (last_hidden_state): {text_features.last_hidden_state.shape}")
print("文本编码完成！")

# 计算相似度
print("\n正在计算图像-文本相似度...")
# 获取归一化后的特征
image_embeds = image_features.pooler_output  # [1, 512]
text_embeds = text_features.pooler_output    # [3, 512]

# 归一化特征
image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

# 计算余弦相似度 (图像 vs 所有文本)
logits_per_image = torch.matmul(image_embeds_norm, text_embeds_norm.T)  # [1, 3]
print(f"图像-文本相似度矩阵: {logits_per_image}")

# 转换为概率
probs = torch.softmax(logits_per_image, dim=-1)
print(f"相似度概率: {probs}")

# 找到最匹配的文本
max_idx = probs.argmax().item()
print(f"最匹配的文本: {texts[max_idx]}")
print("相似度计算完成！")

# 强制退出程序，避免后台线程阻止退出
import sys
sys.exit(0)