# 数据流说明 - 输入输出完全本地化

## 📊 完整数据流图

```
宿主机（你的本地服务器）                     Docker容器（临时计算环境）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────┐
│  /your/project/input/       │  ────映射(只读)────>  ┌─────────────┐
│    protein.json             │                      │ /input/     │
│    (你的输入文件)            │  <────读取────────    │             │
└─────────────────────────────┘                      └─────────────┘
                                                             │
┌─────────────────────────────┐                             │
│  /mnt/nvme/share/           │  ────映射(只读)────>  ┌─────▼─────────┐
│    msa_datasets/            │                      │ /msa/         │
│    (MSA数据库)               │  <────查询────────    │               │
└─────────────────────────────┘                      └───────────────┘
                                                             │
┌─────────────────────────────┐                             │
│  /mnt/nvme/share/ckpt/      │  ────映射(只读)────>  ┌─────▼─────────┐
│    HelixFold3.pdparams      │                      │ /ckpt/        │
│    (模型权重)                │  <────加载────────    │               │
└─────────────────────────────┘                      └───────────────┘
                                                             │
                                                             ▼
                                                      ┌─────────────┐
                                                      │  推理计算    │
                                                      │  inference.py│
                                                      └──────┬──────┘
                                                             │
┌─────────────────────────────┐                             │
│  /your/project/output/      │  <────写入────────    ┌─────▼─────────┐
│    predicted_structures/    │                      │ /output/      │
│    final_features.pkl       │  ────映射(可写)────>  │               │
│    (推理结果)                │                      └───────────────┘
└─────────────────────────────┘

                                    容器运行结束，自动删除 ×
                                    但数据保留在宿主机 ✓
```

## 🎯 关键理解

### 1. 输入文件（完全本地）

**你的操作：**
```bash
# 你在本地准备输入文件
/home/user/my_project/inputs/protein_a.json
/home/user/my_project/inputs/protein_b.json
```

**运行推理：**
```bash
bash run_docker_infer_custom.sh \
    -i /home/user/my_project/inputs/protein_a.json \
    -o /home/user/my_project/outputs
```

**Docker做的事情：**
```bash
# Docker自动映射
宿主机: /home/user/my_project/inputs/  →  容器: /input/
# 容器内程序读取 /input/protein_a.json
# 实际读取的是宿主机的 /home/user/my_project/inputs/protein_a.json
```

### 2. 输出结果（直接写入本地）

**Docker做的事情：**
```bash
# Docker自动映射
宿主机: /home/user/my_project/outputs/  ←  容器: /output/
# 容器内程序写入 /output/protein_a/...
# 实际写入宿主机的 /home/user/my_project/outputs/protein_a/...
```

**结果：**
```bash
# 推理完成后，你可以在本地看到
/home/user/my_project/outputs/
└── protein_a/
    ├── final_features.pkl
    ├── predicted_structures/
    │   ├── model_0.cif
    │   ├── model_1.cif
    │   └── ...
    └── ...
```

## 🔄 完整流程示例

### 场景：预测一个新的蛋白质复合物

**步骤1: 准备输入**
```bash
# 你在本地创建输入文件
/data/my_research/proteins/complex_001.json
```

**步骤2: 运行推理**
```bash
bash run_docker_infer_custom.sh \
    -i /data/my_research/proteins/complex_001.json \
    -o /data/my_research/results \
    -n 10
```

**步骤3: Docker内部发生的事情**
```
1. 启动容器
2. 映射目录：
   /data/my_research/proteins/ → /input/
   /data/my_research/results/  → /output/
   /mnt/nvme/share/msa_datasets/ → /msa/
   /mnt/nvme/share/ckpt/ → /ckpt/
3. 运行 inference.py
   - 读取 /input/complex_001.json (实际: 宿主机文件)
   - 查询 /msa/ (实际: 宿主机数据库)
   - 加载 /ckpt/模型 (实际: 宿主机权重)
   - 计算...
   - 写入 /output/ (实际: 写到宿主机)
4. 容器退出并删除
```

**步骤4: 查看结果**
```bash
# 结果已经在你的本地目录
ls /data/my_research/results/complex_001/
```

## 📁 目录映射对照表

| 宿主机路径 | 容器内路径 | 权限 | 说明 |
|-----------|-----------|------|------|
| 你的输入文件所在目录 | `/input/` | 只读(ro) | 自动根据输入文件确定 |
| 你的输出目录 | `/output/` | 读写 | 命令行 `-o` 参数指定 |
| `/mnt/nvme/share/msa_datasets/` | `/msa/` | 只读(ro) | MSA数据库（固定） |
| `/mnt/nvme/share/ckpt/` | `/ckpt/` | 只读(ro) | 模型权重（固定） |

## ✅ 验证输入输出本地化

### 测试1: 检查输入来自本地

```bash
# 1. 创建一个测试输入文件
echo '{"test": "data"}' > /tmp/test_input.json

# 2. 运行推理（会失败，但能验证路径）
bash run_docker_infer_custom.sh -i /tmp/test_input.json -o /tmp/test_output

# 3. 容器会尝试读取你本地的 /tmp/test_input.json
```

### 测试2: 检查输出保存到本地

```bash
# 1. 指定输出目录
OUTPUT_DIR="/tmp/my_inference_$(date +%s)"

# 2. 运行推理
bash run_docker_infer_custom.sh -i ./data/demo.json -o "$OUTPUT_DIR" -n 1

# 3. 检查输出文件（在你的本地）
ls -lh "$OUTPUT_DIR"
# 你会看到结果文件，证明是写入本地的
```

## 🚫 常见误解澄清

### 误解1: "数据在容器内"
❌ **错误**：数据保存在容器内，需要用 `docker cp` 复制出来
✅ **正确**：数据始终在本地，容器只是借用本地目录

### 误解2: "需要把文件复制到容器"
❌ **错误**：先把输入文件复制到容器内，再运行推理
✅ **正确**：直接指定本地文件路径，Docker自动映射

### 误解3: "容器内有预置的输入文件"
❌ **错误**：使用容器内的示例文件
✅ **正确**：容器内示例文件只是测试用，你用自己本地的文件

### 误解4: "输出目录必须在项目目录下"
❌ **错误**：输出只能在 `./output`
✅ **正确**：输出可以在任何本地目录，如 `/data/results/`, `/home/user/outputs/` 等

## 📝 实际使用场景

### 场景1: 研究项目

```bash
# 你的项目结构（完全本地）
/home/user/research/
├── raw_sequences/
│   ├── seq_001.json
│   ├── seq_002.json
│   └── seq_003.json
└── predictions/
    └── (推理结果将保存在这里)

# 批量推理
for seq in /home/user/research/raw_sequences/*.json; do
    bash run_docker_infer_custom.sh \
        -i "$seq" \
        -o /home/user/research/predictions \
        -n 5
done
```

### 场景2: 协作项目

```bash
# 团队共享目录（NFS/共享盘）
/shared/team_project/
├── inputs/          # 团队成员放入的蛋白质序列
└── results/         # 推理结果

# 任何人都可以运行
bash run_docker_infer_custom.sh \
    -i /shared/team_project/inputs/protein_new.json \
    -o /shared/team_project/results \
    -n 10
```

### 场景3: 临时测试

```bash
# 快速测试一个新序列
bash run_docker_infer_custom.sh \
    -i /tmp/quick_test.json \
    -o /tmp/test_results \
    -n 1

# 查看结果
ls /tmp/test_results/

# 结果不满意？直接删除
rm -rf /tmp/test_results

# 容器早已删除，没有任何遗留
```

## 💡 核心原则

1. **所有用户数据都在宿主机本地**
2. **容器是无状态的计算环境**
3. **容器删除后，你的数据完好无损**
4. **可以使用任意本地路径作为输入输出**
5. **不需要进入容器、复制文件或修改容器内配置**
