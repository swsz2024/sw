#!/bin/bash
# 启动 Docker 容器并运行推理

set -e  # 遇到错误立即退出

echo "=========================================="
echo "HelixFold3 Docker 推理"
echo "=========================================="

# ==================== 配置部分 ====================
# Docker 镜像配置
IMAGE_NAME="helixfold3-mmseqs2:v1.0"
CONTAINER_NAME="helixfold3_inference_$(date +%Y%m%d_%H%M%S)"

# 宿主机路径配置 (根据实际情况修改)
HOST_MSA_PATH="/mnt/nvme/share/msa_datasets"              # MSA数据库路径
HOST_CKPT_PATH="/mnt/nvme/share/ckpt"                     # 模型权重路径
HOST_INPUT="$(pwd)/data"                                   # 输入JSON文件所在目录
HOST_OUTPUT="$(pwd)/output"                                # 输出目录

# 容器内路径 (固定，不要修改)
CONTAINER_MSA_PATH="/msa"
CONTAINER_CKPT_PATH="/ckpt"
CONTAINER_INPUT="/input"
CONTAINER_OUTPUT="/output"

# GPU 配置
GPU_DEVICE="0"  # 指定使用的GPU设备ID

# 推理参数 (可自定义)
MODEL_CKPT="/ckpt/HelixFold3-240814.pdparams"           # 模型权重文件名
INPUT_JSON="/input/demo_6zcy_smiles.json"              # 输入JSON文件 (容器内路径)
OUTPUT_DIR="/output"                                    # 输出目录 (容器内路径)

# 推理超参数 (可自定义)
INFER_TIMES="1"          # 推理次数
DIFF_BATCH_SIZE="1"     # Diffusion batch size
PRECISION="fp32"         # 精度: fp32 / fp16 / bf16
SEARCH_TOOL="mmseqs"     # 搜索工具: mmseqs / hmmer

# ==================================================

# 检查输出目录
if [ ! -d "$HOST_OUTPUT" ]; then
    echo "创建输出目录: $HOST_OUTPUT"
    mkdir -p "$HOST_OUTPUT"
fi

# 检查输入目录
if [ ! -d "$HOST_INPUT" ]; then
    echo "错误: 输入目录不存在: $HOST_INPUT"
    exit 1
fi

# 检查MSA数据库路径
if [ ! -d "$HOST_MSA_PATH" ]; then
    echo "警告: MSA数据库路径不存在: $HOST_MSA_PATH"
    echo "请检查路径配置是否正确"
fi

# 检查模型权重路径
if [ ! -d "$HOST_CKPT_PATH" ]; then
    echo "警告: 模型权重路径不存在: $HOST_CKPT_PATH"
    echo "请检查路径配置是否正确"
fi

echo ""
echo "配置信息:"
echo "  镜像名称: $IMAGE_NAME"
echo "  容器名称: $CONTAINER_NAME"
echo "  GPU设备: $GPU_DEVICE"
echo "  MSA数据库: $HOST_MSA_PATH -> $CONTAINER_MSA_PATH"
echo "  模型权重: $HOST_CKPT_PATH -> $CONTAINER_CKPT_PATH"
echo "  输入目录: $HOST_INPUT -> $CONTAINER_INPUT"
echo "  输出目录: $HOST_OUTPUT -> $CONTAINER_OUTPUT"
echo ""
echo "推理参数:"
echo "  输入文件: $INPUT_JSON"
echo "  推理次数: $INFER_TIMES"
echo "  Diffusion Batch: $DIFF_BATCH_SIZE"
echo "  精度: $PRECISION"
echo "  搜索工具: $SEARCH_TOOL"
echo ""

# 运行 Docker 容器
echo "启动 Docker 容器..."
echo ""

docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus "device=$GPU_DEVICE" \
    --shm-size=8g \
    -v "$HOST_MSA_PATH:$CONTAINER_MSA_PATH:ro" \
    -v "$HOST_CKPT_PATH:$CONTAINER_CKPT_PATH:ro" \
    -v "$HOST_INPUT:$CONTAINER_INPUT:ro" \
    -v "$HOST_OUTPUT:$CONTAINER_OUTPUT" \
    -e MODEL_CKPT="$MODEL_CKPT" \
    -e INPUT_JSON="$INPUT_JSON" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e INFER_TIMES="$INFER_TIMES" \
    -e DIFF_BATCH_SIZE="$DIFF_BATCH_SIZE" \
    -e PRECISION="$PRECISION" \
    -e SEARCH_TOOL="$SEARCH_TOOL" \
    -e CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    "$IMAGE_NAME" \
    bash /app/swbind/run_infer_docker.sh

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 推理完成！"
    echo "输出目录: $HOST_OUTPUT"
    echo "=========================================="
else
    echo ""
    echo "✗ 推理失败"
    exit 1
fi
