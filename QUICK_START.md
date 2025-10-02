# HelixFold3 Docker 快速开始

## 🎯 核心概念：本地输入输出

**重要**：所有的输入数据和输出结果都在**宿主机本地**，容器只是计算环境！

```
宿主机（你的服务器）          Docker 容器（计算环境）
├─ 本地输入文件              ←映射→  容器内读取
├─ 本地输出目录              ←映射→  容器内写入
├─ MSA数据库 (只读)          ←映射→  容器内读取
└─ 模型权重 (只读)           ←映射→  容器内读取
```

## 📦 步骤1: 构建镜像（只需一次）

```bash
cd /mnt/nvme/lijq/swbind/helixfold3-mmseqs2
bash build_docker.sh
```

## 🚀 步骤2: 运行推理

### 推荐方式：使用命令行参数

```bash
# 基本用法：指定本地输入文件和输出目录
bash run_docker_infer_custom.sh \
    -i /path/to/your/local/protein.json \
    -o /path/to/your/local/output

# 自定义推理次数
bash run_docker_infer_custom.sh \
    -i /path/to/your/local/protein.json \
    -o /path/to/your/local/output \
    -n 6

# 完整示例
bash run_docker_infer_custom.sh \
    -i /home/user/my_data/protein_complex.json \
    -o /home/user/my_results \
    -n 10 \
    -b 4 \
    -g 1
```

## 💡 实际使用示例

### 示例1: 单个蛋白质推理

假设你有一个蛋白质结构预测任务：

```bash
# 你的输入文件在
/home/lijq/projects/protein_study/input/my_protein.json

# 你想把结果保存到
/home/lijq/projects/protein_study/results/

# 运行命令
bash run_docker_infer_custom.sh \
    -i /home/lijq/projects/protein_study/input/my_protein.json \
    -o /home/lijq/projects/protein_study/results/ \
    -n 5
```

**容器运行时的映射关系：**
- 宿主机 `/home/lijq/projects/protein_study/input/` → 容器内 `/input/`
- 宿主机 `/home/lijq/projects/protein_study/results/` → 容器内 `/output/`
- 容器读取 `/input/my_protein.json`，实际读取的是宿主机的文件
- 容器写入 `/output/...`，实际写入宿主机的目录

### 示例2: 批量处理多个蛋白质

```bash
#!/bin/bash
# 批量处理脚本

INPUT_DIR="/data/proteins/batch_001"
OUTPUT_DIR="/data/results/batch_001"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 遍历所有JSON文件
for json_file in "$INPUT_DIR"/*.json; do
    echo "处理: $json_file"
    
    bash run_docker_infer_custom.sh \
        -i "$json_file" \
        -o "$OUTPUT_DIR" \
        -n 5 \
        -b 2
    
    echo "完成: $(basename $json_file)"
    echo "---"
done

echo "所有任务完成！"
```

### 示例3: 不同GPU和参数配置

```bash
# GPU 0 运行快速预测（少次数）
bash run_docker_infer_custom.sh \
    -i /data/protein_a.json \
    -o /data/results_quick \
    -n 3 \
    -g 0

# GPU 1 运行高质量预测（多次数）
bash run_docker_infer_custom.sh \
    -i /data/protein_b.json \
    -o /data/results_high_quality \
    -n 20 \
    -b 8 \
    -g 1
```

## 📂 文件组织建议

推荐的本地文件组织方式：

```
/your/workspace/
├── inputs/                          # 你的输入文件
│   ├── protein_001.json
│   ├── protein_002.json
│   └── complex_001.json
│
├── outputs/                         # 推理结果输出
│   ├── protein_001/
│   │   ├── final_features.pkl
│   │   ├── predicted_structures/
│   │   └── ...
│   ├── protein_002/
│   └── complex_001/
│
└── batch_run.sh                     # 批量运行脚本
```

运行示例：

```bash
cd /your/workspace

# 单个文件
bash /path/to/run_docker_infer_custom.sh \
    -i inputs/protein_001.json \
    -o outputs

# 批量处理
for f in inputs/*.json; do
    bash /path/to/run_docker_infer_custom.sh -i "$f" -o outputs -n 6
done
```

## 🔍 验证输入输出是否正确

运行前检查：

```bash
# 1. 检查输入文件存在
ls -lh /path/to/your/input.json

# 2. 检查输出目录（会自动创建，但可以提前创建）
mkdir -p /path/to/your/output

# 3. 运行推理
bash run_docker_infer_custom.sh \
    -i /path/to/your/input.json \
    -o /path/to/your/output \
    -n 5

# 4. 检查结果
ls -lh /path/to/your/output/
```

## ⚙️ 所有可用参数

```bash
-i, --input JSON_FILE        输入JSON文件的完整路径（宿主机路径）
-o, --output OUTPUT_DIR      输出目录的完整路径（宿主机路径）
-n, --infer_times N          推理次数（默认: 1）
-b, --batch_size N           Diffusion batch size（默认: 1）
-p, --precision PREC         精度 fp32/fp16/bf16（默认: fp32）
-s, --search_tool TOOL       搜索工具 mmseqs/hmmer（默认: mmseqs）
-g, --gpu DEVICE             GPU设备ID（默认: 0）
-m, --model MODEL_FILE       模型文件名（默认: HelixFold3-240814.pdparams）
-h, --help                   显示帮助信息
```

## 🎓 理解 Docker 映射

```bash
# 这条命令中：
bash run_docker_infer_custom.sh \
    -i /home/user/data/protein.json \
    -o /home/user/results

# Docker 实际执行的映射：
docker run \
    -v "/home/user/data:/input:ro" \      # 本地输入目录 → 容器 /input （只读）
    -v "/home/user/results:/output" \     # 本地输出目录 → 容器 /output （可写）
    ...

# 容器内程序：
# - 从 /input/protein.json 读取（实际是宿主机 /home/user/data/protein.json）
# - 写入 /output/... （实际写到宿主机 /home/user/results/...）
```

**关键点：**
1. ✅ 输入文件完全在你的本地服务器上
2. ✅ 输出结果直接写到你的本地目录
3. ✅ 容器只是运行环境，不保存任何数据
4. ✅ 容器删除后，你的数据完好无损

## 📊 实际数据流

```
[本地输入文件] 
    ↓ (Docker -v 映射，只读)
[容器读取] 
    ↓ (计算)
[容器生成结果] 
    ↓ (Docker -v 映射，写入)
[本地输出目录] ✓ 结果保存在本地
```

## ⭐ 关键提示

1. **容器内的示例文件**：容器内可能有一些示例JSON，但你**不需要使用它们**
2. **完全本地化**：你的所有数据都在本地，容器只是借用来计算
3. **相对路径也支持**：可以使用相对路径，如 `./my_data/protein.json`
4. **自动创建输出目录**：如果输出目录不存在，脚本会自动创建

## 🚨 常见误区

❌ **错误理解**：需要把数据复制到容器内
✅ **正确理解**：直接使用本地路径，Docker自动映射

❌ **错误理解**：输出保存在容器内，需要复制出来
✅ **正确理解**：输出直接写到本地目录，容器删除后数据依然存在

❌ **错误理解**：每次运行需要修改容器内的配置
✅ **正确理解**：直接用命令行参数指定，无需进入容器
