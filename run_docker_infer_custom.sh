#!/bin/bash
# 启动 Docker 容器并运行推理 - 支持命令行参数自定义

set -e  # 遇到错误立即退出

# ==================== 使用说明 ====================
show_usage() {
    cat << EOF
使用方法:
    bash $0 [选项]

选项:
    -i, --input JSON_FILE        输入JSON文件路径 (宿主机路径)
    -o, --output OUTPUT_DIR      输出目录 (宿主机路径，默认: ./output)
    -n, --infer_times N          推理次数 (默认: 1)
    -b, --batch_size N           Diffusion batch size (默认: 1)
    -p, --precision PREC         精度 fp32/fp16/bf16 (默认: fp32)
    -s, --search_tool TOOL       搜索工具 mmseqs/hmmer (默认: mmseqs)
    -g, --gpu DEVICE             GPU设备ID (默认: 0)
    -m, --model MODEL_FILE       模型文件名 (默认: HelixFold3-240814.pdparams)
    -h, --help                   显示此帮助信息

数据库路径配置 (可选，默认使用脚本中的配置):
    --msa_path PATH              MSA数据库路径
    --ckpt_path PATH             模型权重目录路径

示例:
    # 基本使用
    bash $0 -i ./data/demo.json -o ./output

    # 自定义推理次数和batch size
    bash $0 -i ./data/demo.json -o ./output -n 6 -b 2

    # 使用HMMER搜索工具
    bash $0 -i ./data/demo.json -o ./output -s hmmer

    # 完整自定义
    bash $0 -i ./data/my_protein.json -o ./results -n 10 -b 4 -p fp16 -g 1

EOF
}

# ==================== 默认配置 ====================
# Docker 镜像配置
IMAGE_NAME="helixfold3-mmseqs2:v1.0"
CONTAINER_NAME="helixfold3_inference_$(date +%Y%m%d_%H%M%S)"

# 宿主机路径配置 (默认值)
HOST_MSA_PATH="/mnt/nvme/share/msa_datasets"
HOST_CKPT_PATH="/mnt/nvme/share/ckpt"
HOST_INPUT=""
HOST_OUTPUT="$(pwd)/output"

# 容器内路径 (固定)
CONTAINER_MSA_PATH="/msa"
CONTAINER_CKPT_PATH="/ckpt"
CONTAINER_INPUT="/input"
CONTAINER_OUTPUT="/output"

# 推理参数 (默认值)
MODEL_CKPT="/ckpt/HelixFold3-240814.pdparams"
INPUT_JSON=""
INFER_TIMES="1"
DIFF_BATCH_SIZE="1"
PRECISION="fp32"
SEARCH_TOOL="mmseqs"
GPU_DEVICE="0"

# ==================== 解析命令行参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            HOST_INPUT="$2"
            shift 2
            ;;
        -o|--output)
            HOST_OUTPUT="$2"
            shift 2
            ;;
        -n|--infer_times)
            INFER_TIMES="$2"
            shift 2
            ;;
        -b|--batch_size)
            DIFF_BATCH_SIZE="$2"
            shift 2
            ;;
        -p|--precision)
            PRECISION="$2"
            shift 2
            ;;
        -s|--search_tool)
            SEARCH_TOOL="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_CKPT="/ckpt/$2"
            shift 2
            ;;
        --msa_path)
            HOST_MSA_PATH="$2"
            shift 2
            ;;
        --ckpt_path)
            HOST_CKPT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "错误: 未知参数 '$1'"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# ==================== 参数验证 ====================
if [ -z "$HOST_INPUT" ]; then
    echo "错误: 必须指定输入文件 (-i 或 --input)"
    echo ""
    show_usage
    exit 1
fi

if [ ! -f "$HOST_INPUT" ]; then
    echo "错误: 输入文件不存在: $HOST_INPUT"
    exit 1
fi

# 获取输入文件的目录和文件名
INPUT_DIR=$(dirname "$HOST_INPUT")
INPUT_FILENAME=$(basename "$HOST_INPUT")
INPUT_JSON="/input/$INPUT_FILENAME"

# ==================== 目录检查和创建 ====================
echo "=========================================="
echo "HelixFold3 Docker 推理"
echo "=========================================="

# 创建输出目录
if [ ! -d "$HOST_OUTPUT" ]; then
    echo "创建输出目录: $HOST_OUTPUT"
    mkdir -p "$HOST_OUTPUT"
fi

# 检查MSA数据库路径
if [ ! -d "$HOST_MSA_PATH" ]; then
    echo "警告: MSA数据库路径不存在: $HOST_MSA_PATH"
fi

# 检查模型权重路径
if [ ! -d "$HOST_CKPT_PATH" ]; then
    echo "警告: 模型权重路径不存在: $HOST_CKPT_PATH"
fi

# ==================== 显示配置信息 ====================
echo ""
echo "配置信息:"
echo "  镜像名称: $IMAGE_NAME"
echo "  容器名称: $CONTAINER_NAME"
echo "  GPU设备: $GPU_DEVICE"
echo ""
echo "数据路径:"
echo "  MSA数据库: $HOST_MSA_PATH -> $CONTAINER_MSA_PATH"
echo "  模型权重: $HOST_CKPT_PATH -> $CONTAINER_CKPT_PATH"
echo "  输入目录: $INPUT_DIR -> $CONTAINER_INPUT"
echo "  输出目录: $HOST_OUTPUT -> $CONTAINER_OUTPUT"
echo ""
echo "推理参数:"
echo "  输入文件: $HOST_INPUT"
echo "  推理次数: $INFER_TIMES"
echo "  Diffusion Batch: $DIFF_BATCH_SIZE"
echo "  精度: $PRECISION"
echo "  搜索工具: $SEARCH_TOOL"
echo ""

# ==================== 运行 Docker 容器 ====================
echo "启动 Docker 容器..."
echo ""

docker run --rm \
    --name "$CONTAINER_NAME" \
    --gpus "device=$GPU_DEVICE" \
    --shm-size=8g \
    -v "$HOST_MSA_PATH:$CONTAINER_MSA_PATH:ro" \
    -v "$HOST_CKPT_PATH:$CONTAINER_CKPT_PATH:ro" \
    -v "$INPUT_DIR:$CONTAINER_INPUT:ro" \
    -v "$HOST_OUTPUT:$CONTAINER_OUTPUT" \
    -e MODEL_CKPT="$MODEL_CKPT" \
    -e INPUT_JSON="$INPUT_JSON" \
    -e OUTPUT_DIR="$CONTAINER_OUTPUT" \
    -e INFER_TIMES="$INFER_TIMES" \
    -e DIFF_BATCH_SIZE="$DIFF_BATCH_SIZE" \
    -e PRECISION="$PRECISION" \
    -e SEARCH_TOOL="$SEARCH_TOOL" \
    -e CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    "$IMAGE_NAME" \
    bash /app/swbind/run_infer_docker.sh

# ==================== 检查执行结果 ====================
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
