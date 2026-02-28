#!/bin/bash
# 清除所有代理环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# 激活 conda 环境
source /home/baojiali/anaconda3/etc/profile.d/conda.sh
conda activate clip-tutorial

# 再次清除代理（conda activate 可能会重新设置）
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# 运行 Python 脚本
exec python "$@"
