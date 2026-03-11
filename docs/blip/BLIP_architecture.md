# BLIP 原理与网络结构详解

## 1. 从 CLIP 的局限说起

CLIP 是**双塔架构**：

```
Image ──→ [Vision Encoder] ──→ image_embed ─┐
                                             ├──→ cosine similarity
Text  ──→ [Text Encoder]   ──→ text_embed  ─┘
```

两个编码器**各干各的**，最后只通过一个点积来交互。这意味着：
- 图像和文本之间**没有深层交互**（不知道图中哪个区域对应哪个词）
- **只能做匹配/检索**，不能生成文字

BLIP 的设计目标：**一个模型，同时搞定理解和生成。**

---

## 2. BLIP 的三头架构

BLIP 的核心思想是：**共享一个 Vision Encoder，接三个不同的文本模块**，分别对应三个训练目标。

```
                         ┌─────────────────────┐
                         │    Vision Encoder    │
                         │    (ViT-B/16)        │
                         └──────────┬──────────┘
                                    │
                          image features (序列)
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │   Text Encoder   │  │ Image-grounded   │  │ Image-grounded   │
   │                  │  │ Text Encoder      │  │ Text Decoder     │
   │  (无cross-attn)  │  │  (有cross-attn)   │  │  (有cross-attn)  │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │                     │                      │
            ▼                     ▼                      ▼
     ITC Loss              ITM Loss               LM Loss
   (对比学习)            (二分类匹配)           (语言建模)
```

---

### 头 1: ITC (Image-Text Contrastive) —— 和 CLIP 一样

```
image_embed = ViT(image).cls_token        # [1, 768] → 线性投影 → [1, 256]
text_embed  = TextEnc(text).cls_token     # [1, 768] → 线性投影 → [1, 256]

similarity = cosine(image_embed, text_embed)
loss = contrastive_loss(similarity)       # 和 CLIP 的 loss 一模一样
```

- Text Encoder 这里**不看图像**（没有 cross-attention），和 CLIP 的文本编码器一样
- 目的：学会把匹配的图文对拉近，不匹配的推远
- 用途：**图文检索**（快速，因为图文可以离线编码）

---

### 头 2: ITM (Image-Text Matching) —— 比 CLIP 更精细的匹配

```
image_features = ViT(image)               # [1, 197, 768]  (196个patch + 1个cls)
text_features  = TextEnc(text)            # [1, L, 768]

# 关键区别：text encoder 中加入了 cross-attention 层
# 文本的每个 token 都能"看到"图像的每个 patch
fused = CrossAttention(text_features, image_features)

itm_logits = ITM_Head(fused.cls_token)    # [1, 2]  →  匹配 or 不匹配
```

- 和 ITC 的区别：文本编码器中**插入了 cross-attention 层**，文本 token 能直接关注图像 patch
- 这是一个**二分类任务**：这对图文匹配吗？是/否
- 用途：精细的图文匹配（更准但更慢，因为图文必须一起过模型）

代码中的 `outputs.itm_score` 就是这个 `[not_match, match]` 的 logits。

---

### 头 3: LM (Language Modeling) —— CLIP 做不到的事

```
image_features = ViT(image)               # [1, 197, 768]

# Text Decoder（注意：这里是 Decoder，不是 Encoder）
# 用 causal mask（每个词只能看到前面的词）+ cross-attention（能看到图像）
caption = Decoder.generate(
    image_features,
    start_token="[DEC]"
)
# 输出: "a woman and her dog on the beach"
```

- 把 Text Encoder 换成了 **Text Decoder**（加了 causal self-attention mask）
- Decoder 通过 cross-attention 看图像特征，**一个词一个词地生成描述**
- 训练时用标准的**自回归语言建模 loss**（预测下一个词）
- 用途：**Image Captioning、VQA**

---

## 3. 一个关键设计：参数共享

三个文本模块并不是三个独立的网络，它们**大量共享参数**：

```
                    Self-Attention    Cross-Attention    FFN
Text Encoder (ITC)      ✓                 ✗             ✓
Text Encoder (ITM)      ✓(共享)           ✓(独有)        ✓(共享)
Text Decoder (LM)       ✓(共享)           ✓(共享自ITM)   ✓(共享)
```

- Self-Attention 和 FFN 的参数在三个模块间**共享**
- Cross-Attention 层是 ITM 和 LM 独有的（ITC 没有）
- 这样模型参数量不会膨胀到 3 倍，而是只比单个模型大一点

---

## 4. CapFilt —— BLIP 最精彩的创新

以上讲的是模型结构。但 BLIP 论文最大的贡献其实是**训练策略**。

**问题**：网上爬来的图文对质量很差（文不对图、文字是广告等等）

**CapFilt 方案**（Captioning and Filtering）：

```
第一轮：用有噪声的网络数据训练 BLIP
           │
           ▼
┌─────────────────────────────────┐
│  训练好的 BLIP 拆成两个角色：      │
│                                 │
│  Captioner (LM 头)              │
│  → 给图片生成新的、更好的描述      │
│                                 │
│  Filter (ITM 头)                │
│  → 过滤掉不匹配的图文对           │
│  → 也过滤 Captioner 生成的坏描述   │
└─────────────────────────────────┘
           │
           ▼
第二轮：用清洗后的数据重新训练 BLIP
           │
           ▼
       更好的模型！
```

简单说就是：**模型自己给自己造数据、自己审核质量、然后用更好的数据再训练自己。**

这就是 "Bootstrapped"（自举）的含义。

---

## 5. 总结对比

| | CLIP | BLIP |
|---|---|---|
| Vision Encoder | ViT 或 ResNet | ViT |
| 文本端 | 1个 Text Encoder | 3个模块（共享参数） |
| 图文交互 | 余弦相似度（浅） | ITC(浅) + ITM/LM(cross-attention, 深) |
| 能力 | 匹配、检索、零样本分类 | 匹配 + 检索 + 描述生成 + VQA |
| 训练数据 | 4亿图文对（硬扛噪声） | CapFilt 自动清洗数据 |
| 参数量(base) | ~151M | ~224M |

**一句话理解 BLIP**：在 CLIP 的对比学习基础上，加了 cross-attention 做精细匹配，加了 decoder 做文本生成，再用自举策略解决数据质量问题。
