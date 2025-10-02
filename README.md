# 此目录为paddle版Helixfold3的推理代码，同时适配hmmer和mmseqs2搜索工具


1. 运行前需要激活环境：
```source ../helixfold3-main/.venv/bin/activate```

2. 更改 run_infer.sh
    1. INPUT_JSON=${2:-"data/demo_\<xxxx>.json"}
    2. --search_tool : hmmer | mmseqs

3. 运行
```bash run_mmseqs_infer.sh```或```bash run_hmmer_infer.sh```

4. 改动
    1. 添加了mmseqs2的适配
    2. 部分文件import helixfold改为import src.helixfold, common和model目录下的除外


