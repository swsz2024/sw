# HelixFold3 Docker 使用指南

## 📦 构建 Docker 镜像

```bash
cd /mnt/nvme/lijq/swbind/helixfold3-mmseqs2
bash build_docker.sh
```

构建过程会：
1. 自动打包本地的 HMMER 和 MMseqs2 工具
2. 构建包含所有依赖的 Docker 镜像
3. 清理临时文件

## 🚀 运行推理

有两种方式运行推理：

### 方式1: 使用默认配置（简单）

编辑 `run_docker_infer.sh` 修改配置，然后运行：

```bash
bash run_docker_infer.sh
```

### 方式2: 使用命令行参数（灵活，推荐）

```bash
bash run_docker_infer_custom.sh -i <输入文件> [选项]
```

#### 常用示例

**基本使用：**
```bash
bash run_docker_infer_custom.sh -i ./data/demo_6zcy_smiles.json -o ./output
```

**推理6次：**
```bash
bash run_docker_infer_custom.sh -i ./data/my_protein.json -o ./output -n 6
```

**推理10次，batch size 4：**
```bash
bash run_docker_infer_custom.sh -i ./data/my_protein.json -o ./output -n 10 -b 4
```

**使用 HMMER 搜索工具：**
```bash
bash run_docker_infer_custom.sh -i ./data/my_protein.json -o ./output -s hmmer
```

**使用 GPU 1，fp16 精度：**
```bash
bash run_docker_infer_custom.sh -i ./data/my_protein.json -o ./output -g 1 -p fp16
```

**完整自定义：**
```bash
bash run_docker_infer_custom.sh \
    -i ./data/my_protein.json \
    -o ./results \
    -n 10 \
    -b 4 \
    -p fp16 \
    -g 1 \
    -s mmseqs
```

#### 所有可用选项

```
-i, --input JSON_FILE        输入JSON文件路径 (必需)
-o, --output OUTPUT_DIR      输出目录 (默认: ./output)
-n, --infer_times N          推理次数 (默认: 1)
-b, --batch_size N           Diffusion batch size (默认: 1)
-p, --precision PREC         精度 fp32/fp16/bf16 (默认: fp32)
-s, --search_tool TOOL       搜索工具 mmseqs/hmmer (默认: mmseqs)
-g, --gpu DEVICE             GPU设备ID (默认: 0)
-m, --model MODEL_FILE       模型文件名 (默认: HelixFold3-240814.pdparams)
-h, --help                   显示帮助信息

数据库路径配置 (可选):
--msa_path PATH              MSA数据库路径 (默认: /mnt/nvme/share/msa_datasets)
--ckpt_path PATH             模型权重目录路径 (默认: /mnt/nvme/share/ckpt)
```

## 📂 目录结构说明

### 宿主机目录
- `/mnt/nvme/share/msa_datasets/` - MSA 数据库（只读挂载）
- `/mnt/nvme/share/ckpt/` - 模型权重（只读挂载）
- `./data/` - 输入 JSON 文件
- `./output/` - 推理输出结果

### 容器内映射
- `/msa/` ← 映射自宿主机 MSA 数据库
- `/ckpt/` ← 映射自宿主机模型权重
- `/input/` ← 映射自输入文件所在目录
- `/output/` ← 映射自输出目录

## 🔧 高级配置

### 修改数据库路径

如果你的数据库路径不同，可以使用 `--msa_path` 和 `--ckpt_path` 参数：

```bash
bash run_docker_infer_custom.sh \
    -i ./data/my_protein.json \
    -o ./output \
    --msa_path /your/custom/msa/path \
    --ckpt_path /your/custom/ckpt/path
```

或者编辑 `run_docker_infer.sh` 或 `run_docker_infer_custom.sh` 中的默认路径：

```bash
HOST_MSA_PATH="/your/custom/msa/path"
HOST_CKPT_PATH="/your/custom/ckpt/path"
```

### 修改默认推理参数

编辑 `run_docker_infer.sh` 中的参数：

```bash
# 推理超参数 (可自定义)
INFER_TIMES="6"          # 推理次数
DIFF_BATCH_SIZE="4"      # Diffusion batch size
PRECISION="fp16"         # 精度: fp32 / fp16 / bf16
SEARCH_TOOL="hmmer"      # 搜索工具: mmseqs / hmmer
```

## 📊 输出结果

推理完成后，结果会保存在指定的输出目录中：

```
output/
└── <input_name>/
    ├── final_features.pkl
    ├── predicted_structures/
    └── ...
```

## ⚠️ 注意事项

1. **首次运行**：首次运行时会初始化数据库，可能需要较长时间
2. **GPU 内存**：确保 GPU 有足够内存（推荐 24GB+）
3. **推理次数**：增加推理次数会显著增加运行时间
4. **数据保护**：数据库和权重以只读模式挂载，不会被修改

## 🐛 故障排查

### 问题：找不到数据库
检查数据库路径配置是否正确：
```bash
ls /mnt/nvme/share/msa_datasets/
ls /mnt/nvme/share/ckpt/
```

### 问题：GPU 不可用
确保安装了 nvidia-docker2：
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 问题：权限错误
确保脚本有执行权限：
```bash
chmod +x build_docker.sh run_docker_infer.sh run_docker_infer_custom.sh
```

## 📝 文件说明

- `Dockerfile` - Docker 镜像构建文件
- `build_docker.sh` - 构建镜像脚本
- `run_docker_infer.sh` - 默认配置推理脚本
- `run_docker_infer_custom.sh` - 命令行参数推理脚本（推荐）
- `run_infer_docker.sh` - 容器内推理脚本（不需要直接运行）
- `.dockerignore` - Docker 构建忽略文件
