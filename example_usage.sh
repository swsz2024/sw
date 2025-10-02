#!/bin/bash
# HelixFold3 Docker 推理使用示例
# 此脚本展示如何使用本地输入输出进行推理

echo "========================================"
echo "HelixFold3 Docker 使用示例"
echo "========================================"
echo ""

# ==================== 配置你的路径 ====================
# 修改以下路径为你的实际路径

# 你的输入JSON文件（本地路径）
MY_INPUT_FILE="./data/demo_6zcy_smiles.json"

# 你想保存结果的目录（本地路径）
MY_OUTPUT_DIR="./output"

# 推理参数
MY_INFER_TIMES=5        # 推理次数
MY_BATCH_SIZE=1         # Batch size
MY_GPU=0                # 使用的GPU ID

# ==================== 示例1: 基本使用 ====================
echo "示例1: 基本推理（1次）"
echo "输入: $MY_INPUT_FILE"
echo "输出: $MY_OUTPUT_DIR"
echo ""

# 检查输入文件是否存在
if [ ! -f "$MY_INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件不存在: $MY_INPUT_FILE"
    echo "提示: 请修改 MY_INPUT_FILE 变量为你的实际文件路径"
    exit 1
fi

# 运行推理
echo "运行命令:"
echo "bash run_docker_infer_custom.sh -i $MY_INPUT_FILE -o $MY_OUTPUT_DIR"
echo ""
echo "按 Enter 继续，或 Ctrl+C 取消..."
read

bash run_docker_infer_custom.sh \
    -i "$MY_INPUT_FILE" \
    -o "$MY_OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推理完成！"
    echo "结果保存在: $MY_OUTPUT_DIR"
    echo ""
    echo "查看结果:"
    ls -lh "$MY_OUTPUT_DIR"
fi

# ==================== 示例2: 自定义参数 ====================
echo ""
echo "========================================"
echo "示例2: 自定义推理参数（多次推理）"
echo "========================================"
echo "输入: $MY_INPUT_FILE"
echo "输出: ${MY_OUTPUT_DIR}_custom"
echo "推理次数: $MY_INFER_TIMES"
echo "Batch size: $MY_BATCH_SIZE"
echo ""
echo "按 Enter 继续，或 Ctrl+C 取消..."
read

bash run_docker_infer_custom.sh \
    -i "$MY_INPUT_FILE" \
    -o "${MY_OUTPUT_DIR}_custom" \
    -n "$MY_INFER_TIMES" \
    -b "$MY_BATCH_SIZE" \
    -g "$MY_GPU"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 自定义参数推理完成！"
    echo "结果保存在: ${MY_OUTPUT_DIR}_custom"
fi

# ==================== 示例3: 批量处理 ====================
echo ""
echo "========================================"
echo "示例3: 批量处理多个文件"
echo "========================================"
echo "此示例展示如何批量处理多个蛋白质"
echo ""

# 假设你有多个JSON文件在 data 目录
BATCH_INPUT_DIR="./data"
BATCH_OUTPUT_DIR="./output_batch"

if [ -d "$BATCH_INPUT_DIR" ]; then
    JSON_COUNT=$(ls "$BATCH_INPUT_DIR"/*.json 2>/dev/null | wc -l)
    
    if [ $JSON_COUNT -gt 0 ]; then
        echo "找到 $JSON_COUNT 个JSON文件"
        echo "输入目录: $BATCH_INPUT_DIR"
        echo "输出目录: $BATCH_OUTPUT_DIR"
        echo ""
        echo "是否继续批量处理？(y/N)"
        read -r response
        
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            mkdir -p "$BATCH_OUTPUT_DIR"
            
            for json_file in "$BATCH_INPUT_DIR"/*.json; do
                echo ""
                echo "处理: $(basename $json_file)"
                
                bash run_docker_infer_custom.sh \
                    -i "$json_file" \
                    -o "$BATCH_OUTPUT_DIR" \
                    -n 3
                
                echo "完成: $(basename $json_file)"
                echo "---"
            done
            
            echo ""
            echo "✅ 批量处理完成！"
            echo "所有结果保存在: $BATCH_OUTPUT_DIR"
        else
            echo "取消批量处理"
        fi
    else
        echo "在 $BATCH_INPUT_DIR 中没有找到JSON文件"
    fi
else
    echo "目录不存在: $BATCH_INPUT_DIR"
fi

# ==================== 完成 ====================
echo ""
echo "========================================"
echo "示例完成！"
echo "========================================"
echo ""
echo "提示："
echo "1. 你的输入文件始终在本地: $MY_INPUT_FILE"
echo "2. 你的输出结果始终在本地: $MY_OUTPUT_DIR"
echo "3. Docker容器只是计算环境，不保存数据"
echo "4. 可以随时修改此脚本中的路径和参数"
echo ""
echo "更多使用方法，查看: QUICK_START.md"
