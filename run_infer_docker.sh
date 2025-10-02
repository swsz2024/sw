#!/bin/bash
# Docker容器内的推理脚本

# 容器内的路径 (固定)
HMMER_MAS_PATH="/msa/origin"
MMSEQS_MSA_PATH="/msa/mmseqs_db"
BAIDU_PATH="/msa/origin/pdb/baidu"
CKPT_PATH="/ckpt"
PY_VENV=/swbind_venv
PY_VENV_NVIDIA=$PY_VENV/lib/python3.9/site-packages/nvidia

# 搜索工具路径 (容器内固定路径)
HMMER_PATH=/hmmer/bin
MMSEQS_PATH=/mmseqs/bin

# 环境路径配置 HMMER / MMSEQS
export PATH=$HMMER_PATH:$PATH
export PATH=$MMSEQS_PATH:$PATH

# Paddle库路径配置
export LD_LIBRARY_PATH=$PY_VENV_NVIDIA/cudnn/lib/:$PY_VENV_NVIDIA/cublas/lib/:$LD_LIBRARY_PATH

# 推理参数 (通过环境变量传入，提供默认值)
MODEL_CKPT=${MODEL_CKPT:-"$CKPT_PATH/HelixFold3-240814.pdparams"}
INPUT_JSON=${INPUT_JSON:-"/input/demo_6zcy_smiles.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"/output"}
INFER_TIMES=${INFER_TIMES:-"1"}
DIFF_BATCH_SIZE=${DIFF_BATCH_SIZE:-"1"}
PRECISION=${PRECISION:-"fp32"}
SEARCH_TOOL=${SEARCH_TOOL:-"mmseqs"}

# 打印推理配置
echo "==========================================="
echo "HelixFold3 推理配置"
echo "==========================================="
echo "模型: $MODEL_CKPT"
echo "输入: $INPUT_JSON"
echo "输出: $OUTPUT_DIR"
echo "推理次数: $INFER_TIMES"
echo "Diffusion Batch: $DIFF_BATCH_SIZE"
echo "精度: $PRECISION"
echo "搜索工具: $SEARCH_TOOL"
echo "==========================================="
echo ""

# 运行推理
python /app/swbind/inference.py \
    --preset='reduced_dbs' \
    --bfd_database_path "$HMMER_MAS_PATH/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
    --small_bfd_database_path "$MMSEQS_MSA_PATH/small_bfd/bfd-first_non_consensus_sequences" \
    --uniclust30_database_path "$HMMER_MAS_PATH/uniclust30/uniclust30_2018_08" \
    --uniprot_database_path "$MMSEQS_MSA_PATH/uniprot/uniprot_baidu" \
    --pdb_seqres_database_path "$BAIDU_PATH/pdb_seqres.txt" \
    --uniref90_database_path "$MMSEQS_MSA_PATH/uniref90/uniref90_baidu" \
    --mgnify_database_path "$MMSEQS_MSA_PATH/mgnify/mgy_clusters_2018_12" \
    --template_mmcif_dir "$BAIDU_PATH/mmcif_files" \
    --obsolete_pdbs_path "$BAIDU_PATH/obsolete.dat" \
    --ccd_preprocessed_path "$MMSEQS_MSA_PATH/../preprocess/ccd_preprocessed_etkdg.pkl" \
    --rfam_database_path "$HMMER_MAS_PATH/rfam/rfam_14_9_rep_seq.fasta" \
    --init_model "$MODEL_CKPT" \
    --max_template_date=2020-05-14 \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --model_name allatom_demo \
    --infer_times "$INFER_TIMES" \
    --diff_batch_size "$DIFF_BATCH_SIZE" \
    --precision "$PRECISION" \
    --search_tool "$SEARCH_TOOL" \
    --mmseqs2_binary_path "$MMSEQS_PATH/mmseqs" \
    --hmmbuild_binary_path "$HMMER_PATH/hmmbuild" \
    --hmmsearch_binary_path "$HMMER_PATH/hmmsearch"