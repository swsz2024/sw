#!/bin/bash
# Docker镜像构建脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始构建 HelixFold3 Docker 镜像"
echo "=========================================="

# 定义变量
HMMER_LOCAL_PATH="/mnt/nvme/lijq/hmmer"
MMSEQS_LOCAL_PATH="/mnt/nvme/lijq/swbind/mmseqs"
PROJECT_DIR="/mnt/nvme/lijq/swbind/helixfold3-mmseqs2"
IMAGE_NAME="helixfold3-mmseqs2"
IMAGE_TAG="v1.0"

# 切换到项目目录
cd $PROJECT_DIR

# 步骤1: 打包 HMMER
echo ""
echo "步骤 1/4: 打包 HMMER..."
if [ -d "$HMMER_LOCAL_PATH" ]; then
    tar -czf hmmer.tar.gz -C $(dirname $HMMER_LOCAL_PATH) $(basename $HMMER_LOCAL_PATH)
    echo "✓ HMMER 打包完成: hmmer.tar.gz"
else
    echo "✗ 错误: HMMER 目录不存在: $HMMER_LOCAL_PATH"
    exit 1
fi

# 步骤2: 打包 MMseqs2
echo ""
echo "步骤 2/4: 打包 MMseqs2..."
if [ -d "$MMSEQS_LOCAL_PATH" ]; then
    tar -czf mmseqs.tar.gz -C $(dirname $MMSEQS_LOCAL_PATH) $(basename $MMSEQS_LOCAL_PATH)
    echo "✓ MMseqs2 打包完成: mmseqs.tar.gz"
else
    echo "✗ 错误: MMseqs2 目录不存在: $MMSEQS_LOCAL_PATH"
    exit 1
fi

# 步骤3: 检查 Dockerfile 是否存在
echo ""
echo "步骤 3/4: 检查 Dockerfile..."
if [ ! -f "Dockerfile" ]; then
    echo "✗ 错误: Dockerfile 不存在"
    exit 1
fi
echo "✓ Dockerfile 检查完成"

# 步骤4: 构建 Docker 镜像
echo ""
echo "步骤 4/4: 构建 Docker 镜像..."
echo "镜像名称: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Docker 镜像构建成功！"
    echo "镜像名称: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "=========================================="
    echo ""
    
    # 清理临时压缩包
    echo "清理临时文件..."
    rm -f hmmer.tar.gz mmseqs.tar.gz
    echo "✓ 临时文件清理完成"
    echo ""
    
    echo "查看镜像信息:"
    docker images | grep ${IMAGE_NAME}
    echo ""
    echo "提示: 使用以下命令运行推理:"
    echo "  bash run_docker_infer.sh"
else
    echo ""
    echo "✗ Docker 镜像构建失败"
    exit 1
fi
