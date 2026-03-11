import os
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# 创建缓存目录
# =========================
CACHE_DIR = "./cache"
IMAGE_DIR = os.path.join(CACHE_DIR, "images")
MODEL_DIR = os.path.join(CACHE_DIR, "models")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 下载函数（带缓存）
# =========================
def download_if_needed(url, path):
    if os.path.exists(path):
        print(f"Using cached file: {path}")
        return path

    print(f"Downloading: {url}")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()

        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        print(f"Saved to: {path}")
        return path

    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.exists(path):
            print("Using existing cached file.")
            return path
        else:
            raise RuntimeError("File not available locally and download failed.")


# =========================
# 图片URL
# =========================
url1 = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

img1_path = os.path.join(IMAGE_DIR, "demo.jpg")
img2_path = os.path.join(IMAGE_DIR, "cat.jpg")

# 下载或使用缓存
download_if_needed(url1, img1_path)
download_if_needed(url2, img2_path)

# 读取图片
image1 = Image.open(img1_path).convert("RGB")
image2 = Image.open(img2_path).convert("RGB")

# 显示图片
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image1)
axes[0].set_title("Image 1")
axes[0].axis("off")

axes[1].imshow(image2)
axes[1].set_title("Image 2")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# =========================
# 加载 BLIP 模型
# =========================
caption_model_name = "Salesforce/blip-image-captioning-base"
model_cache = os.path.join(MODEL_DIR, "blip-caption")

print(f"正在加载模型: {caption_model_name}")

caption_processor = BlipProcessor.from_pretrained(
    caption_model_name,
    cache_dir=model_cache
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    caption_model_name,
    cache_dir=model_cache
).to(device)

caption_model.eval()

print("模型加载完成！")
print(f"模型参数量: {sum(p.numel() for p in caption_model.parameters()) / 1e6:.1f}M")


# =========================
# 无条件生成
# =========================
for i, image in enumerate([image1, image2], 1):

    inputs = caption_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(**inputs, max_length=50)

    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"Image {i}: {caption}")


# =========================
# 条件生成
# =========================
prompts = [
    "a photography of",
    "this is a scene where",
    "in this image, we can see",
]

print("=== Image 1: 海滩场景 ===")

for prompt in prompts:

    inputs = caption_processor(
        images=image1,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(**inputs, max_length=50)

    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Output: {caption}\n")
    
    
inputs = caption_processor(images=image1, return_tensors="pt").to(device)

with torch.no_grad():
    # 1) Greedy：最快，但结果单一
    greedy_ids = caption_model.generate(**inputs, max_length=50, num_beams=1, do_sample=False)

    # 2) Beam Search：质量更高
    beam_ids = caption_model.generate(**inputs, max_length=50, num_beams=5, do_sample=False)

    # 3) Sampling：每次结果可能不同
    sample_ids = caption_model.generate(
        **inputs, max_length=50, do_sample=True,
        temperature=0.7, top_k=50, top_p=0.95,
    )

print("Greedy:      ", caption_processor.decode(greedy_ids[0], skip_special_tokens=True))
print("Beam Search: ", caption_processor.decode(beam_ids[0], skip_special_tokens=True))
print("Sampling:    ", caption_processor.decode(sample_ids[0], skip_special_tokens=True))

from transformers import BlipForQuestionAnswering

vqa_model_name = "Salesforce/blip-vqa-base"
print(f"正在加载 VQA 模型: {vqa_model_name}")

vqa_processor = BlipProcessor.from_pretrained(vqa_model_name)
vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_name).to(device)
vqa_model.eval()

print("VQA 模型加载完成！")

# 对 image1（海滩场景）提问
questions = [
    "What is the woman doing?",
    "What animal is in the picture?",
    "Where are they?",
    "What color is the sky?",
    "How many people are in the image?",
]

print("=== 对 Image 1 提问 ===\n")
for q in questions:
    inputs = vqa_processor(images=image1, text=q, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = vqa_model.generate(**inputs, max_length=20)

    answer = vqa_processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Q: {q}")
    print(f"  A: {answer}\n")
    
# 对 image2（猫）也试试
cat_questions = [
    "What animal is this?",
    "What is the cat doing?",
    "Is the cat fat?",
]

print("=== 对 Image 2 提问 ===\n")
for q in cat_questions:
    inputs = vqa_processor(images=image2, text=q, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = vqa_model.generate(**inputs, max_length=20)

    answer = vqa_processor.decode(output_ids[0], skip_special_tokens=True)
    print(f"  Q: {q}")
    print(f"  A: {answer}\n")
    
from transformers import BlipForImageTextRetrieval

itm_model_name = "Salesforce/blip-itm-base-coco"
print(f"正在加载 ITM 模型: {itm_model_name}")

itm_processor = BlipProcessor.from_pretrained(itm_model_name)
itm_model = BlipForImageTextRetrieval.from_pretrained(itm_model_name).to(device)
itm_model.eval()

print("ITM 模型加载完成！")

# 测试不同文本和 image1（海滩女人+狗）的匹配程度
candidates = [
    "a woman and a dog on the beach",       # 正确描述
    "a woman sitting with her dog",          # 部分正确
    "a cat sleeping on a bed",              # 完全不相关
    "a dog playing in the ocean",            # 部分相关
    "two men playing basketball",            # 完全不相关
]

print("=== Image 1 与不同文本的匹配度 ===\n")
for text in candidates:
    inputs = itm_processor(images=image1, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = itm_model(**inputs)

    # itm_score: [not_match, match] 的 logits
    itm_scores = outputs.itm_score
    match_prob = torch.softmax(itm_scores, dim=1)[0, 1].item()

    bar = "█" * int(match_prob * 30) + "░" * (30 - int(match_prob * 30))
    print(f"  {match_prob:.1%} {bar}  \"{text}\"")
    
print("=" * 50)
print("  BLIP 三大能力演示 —— Image 2 (猫)")
print("=" * 50)

# 1. Captioning: 描述这张图
inputs = caption_processor(images=image2, return_tensors="pt").to(device)
with torch.no_grad():
    cap_ids = caption_model.generate(**inputs, max_length=50, num_beams=5)
caption = caption_processor.decode(cap_ids[0], skip_special_tokens=True)
print(f"\n📝 Captioning: {caption}")

# 2. VQA: 回答关于这张图的问题
question = "What color is the cat?"
inputs = vqa_processor(images=image2, text=question, return_tensors="pt").to(device)
with torch.no_grad():
    ans_ids = vqa_model.generate(**inputs, max_length=20)
answer = vqa_processor.decode(ans_ids[0], skip_special_tokens=True)
print(f"\n❓ VQA: \"{question}\" → \"{answer}\"")

# 3. ITM: 这段文字和图片匹配吗？
text_match = "a fluffy cat sitting"
text_nomatch = "a dog running in the park"
for text in [text_match, text_nomatch]:
    inputs = itm_processor(images=image2, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        itm_out = itm_model(**inputs)
    prob = torch.softmax(itm_out.itm_score, dim=1)[0, 1].item()
    print(f"\n🔗 ITM: \"{text}\" → match {prob:.1%}")