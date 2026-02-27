# VLM Tutorial

Vision Language Model (VLM) 学习项目

## 项目结构

```
vlm_tutorial/
├── notebooks/          # Jupyter notebooks 学习笔记和实验
├── src/               # 源代码
│   ├── models/        # VLM 模型实现
│   ├── datasets/      # 数据集加载和处理
│   └── utils/         # 工具函数
├── scripts/           # 训练和推理脚本
└── tests/             # 单元测试
```

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 学习路径

1. **基础概念** - 理解 VLM 的基本架构
2. **预训练模型使用** - 学习如何使用 Hugging Face 的预训练模型
3. **模型微调** - 在自定义数据集上微调 VLM
4. **高级主题** - 视频语言模型、多模态大语言模型等

## 主要技术栈

- PyTorch
- Hugging Face Transformers
- Datasets
- Pillow / OpenCV
