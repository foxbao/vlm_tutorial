# CLIP 网络架构

**Contrastive Language-Image Pre-training**

## 架构图

```mermaid
flowchart TD
    %% 定义样式
    classDef vision fill:#4A90E2,stroke:#2E6CB8,color:#FFFFFF;
    classDef text fill:#50E3C2,stroke:#2BA887,color:#FFFFFF;
    classDef shared fill:#F5A623,stroke:#D68910,color:#FFFFFF;
    classDef input fill:#E0E0E0,stroke:#999999,color:#333333;
    classDef output fill:#9B59B6,stroke:#8E44AD,color:#FFFFFF;

    %% 输入数据
    Input["输入数据<br/>(Input Data)"]
    image_input["Image Input<br/>224×224 RGB"]
    text_input["Text Token IDs<br/>最大77 tokens"]

    %% 左侧 - 视觉分支
    Vision["视觉分支 (Vision)"]
    Vision_encoder[" vision_model<br/>ViT-Base/Patch32"]
    Vision_layers[" Vision Transformer<br/>12 Layers"]
    Patch_embed[" Patch Embedding<br/>Linear 3×224 → 192×56"]
    Vision_tokens[" Patch Tokens<br/>56 × 192-dim"]
    CLS_token[" CLS Token<br/>Start of text token"]
    Vision_norm[" LayerNorm"]
    Vision_mlp[" MLP Projector<br/>GeLU + Dropout"]
    Vision_features[" Visual Features<br/>512-dim"]

    %% 右侧 - 文本分支
    Text["文本分支 (Text)"]
    Text_encoder[" text_model<br/>Transformer"]
    Text_layers[" Text Transformer<br/>12 Layers"]
    Text_embed[" Text Embeddings<br/>Token Embedding + Position"]
    Text_tokens[" Text Tokens<br/>Tokens up to 77"]
    Text_norm[" LayerNorm"]
    Text_mlp[" MLP Projector<br/>GeLU + Dropout"]
    Text_features[" Text Features<br/>512-dim"]

    %% 中间共享层
    Shared["共享层 (Shared)"]
    Concat[" Concatenate<br/>[CLS + 56 patches + 20 text pads]"]
    Reshape[" Reshape<br/>ViT: 192×56 → 512<br/>LM: 512 × 文本长度"]
    Linear[" Linear Projection<br/>512 × 512"]
    Shared_features[" Shared Features<br/>512-dim (v_i, w_j)"]

    %% 输出
    Output["输出 (Output)"]
    Logits[" Logits Matrix<br/>VocabSize × 512"]
    Loss[" Negative Log-Likelihood Loss<br/>NLL Loss - Cross Entropy"]
    Temperature[" Temperature Scaling<br/>0.07"]

    %% 连接线
    Input -->|1. Data Loading| image_input
    Input -->|1. Data Loading| text_input

    image_input -->|2. Preprocess| Patch_embed
    Patch_embed -->|-| Vision_tokens
    Vision_tokens -->|3. Vision Transformer| Vision_layers
    Vision_layers -->|-| Vision_norm
    Vision_norm -->|-| Vision_mlp
    Vision_mlp -->|4. Visual Features| Vision_features

    text_input -->|2. Tokenize| Text_embed
    Text_embed -->|3. Text Transformer| Text_layers
    Text_layers -->|4. Text Features| Text_features

    %% 绘制 ViT 架构（详细）
    Patch_embed -->|56 patches| ViT_detail

    subgraph ViT_layers["ViT Layer Details"]
        direction TB
        ViT_L12["ViT Layer 12"] --> ViT_L11["ViT Layer 11"]
        ViT_L11 --> ViT_L10["ViT Layer 10"]
        ViT_L10 --> ViT_L9["ViT Layer 9"]
        ViT_L9 --> ViT_L8["ViT Layer 8"]
        ViT_L8 --> ViT_L7["ViT Layer 7"]
        ViT_L7 --> ViT_L6["ViT Layer 6"]
        ViT_L6 --> ViT_L5["ViT Layer 5"]
        ViT_L5 --> ViT_L4["ViT Layer 4"]
        ViT_L4 --> ViT_L3["ViT Layer 3"]
        ViT_L3 --> ViT_L2["ViT Layer 2"]
        ViT_L2 --> ViT_L1["ViT Layer 1"]
    end

    ViT_L1 --> Vision_features

    %% MLP 拉平
    Vision_features -->|Flatten & Reshape| Shared_features
    Text_features -->|Flatten & Reshape| Shared_features

    %% 共享输出
    Shared_features -->|Compute dot products| Logits
    Logits -->|Temperature scaling| Temperature

    Temperature -->|5. Compute Loss| Loss
    Loss -->|6. Backpropagation| Vision_mlp
    Loss -->|6. Backpropagation| Text_mlp

    %% 样式应用
    class Vision,vision;
    class Text,text;
    class Shared,shared;
    class input,Output,output;
```

## 关键参数

| 组件 | 参数量 | 说明 |
|------|--------|------|
| **Vision Encoder** | 87.6M | ViT-Base with Patch32 |
| **Text Encoder** | 63.8M | Transformer LM |
| **总计** | 151M | CLIP-B/32 模型 |

## 输入/输出维度

- **图像输入**: 224×224 RGB 图像
- **文本输入**: Tokenized text, 最大 77 tokens (包括 CLS + EOS)
- **Embedding 维度**: 512
- **词汇表大小**: 49,408 (CLIP-B/32)
- **Patch 数量**: 56 (224² ÷ 32²)
- **Patch 维度**: 192

## 训练目标

**对比学习 (Contrastive Learning)**

给定的 (图像, 文本) 对批次：

1. **编码**: 视觉编码器 → visual embeddings, 文本编码器 → text embeddings
2. **投影**: 共享的 MLP 投影到 512 维特征空间
3. **计算**: logits_{i,j} = (v_i · w_j)^T / temperature
4. **CLS 位置**: i=0 给出匹配对的正例概率
5. **负例**: 所有其他位置都是负例
6. **损失**: 交叉熵损失，最大化正确对的概率

## 架构特点

- **双分支编码器**: 分离的视觉和文本编码器
- **对比预训练**: 学习图像-文本对齐
- **零样本迁移**: 可用于图像到文本、文本到图像检索
- **ViT Transformer**: 全局注意力，处理整个图像作为序列

## 参考实现

- **论文**: OpenAI CLIP (Radford et al., 2021)
- **代码**: `transformers.ClipModel`, `clip_train.py`, `clip_test.py`