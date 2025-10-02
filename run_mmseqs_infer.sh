#!/bin/bash

HMMER_MAS_PATH="/mnt/nvme/share/msa_datasets/origin"
MMSEQS_MSA_PATH="/mnt/nvme/share/msa_datasets/mmseqs_db"
BAIDU_PATH="/mnt/nvme/share/msa_datasets/origin/pdb/baidu"
CKPT_PATH="/mnt/nvme/share/ckpt"
PY_VENV=/mnt/nvme/lijq/swbind/helixfold3-mmseqs2/.venv
PY_VENV_NVIDIA=$PY_VENV/lib/python3.9/site-packages/nvidia

# search tool path
HMMER_PATH=/mnt/nvme/lijq/hmmer/bin
MMSEQS_PATH=/mnt/nvme/lijq/swbind/mmseqs/bin

# environ path HMMER / MMSEQS
export PATH=$HMMER_PATH:$PATH
export PATH=$MMSEQS_PATH:$PATH

## Paddle can only find libcudnn.so, while there are only libcudnn.so.x via nvidia-cudnn pip installation
## So we should link it somewhere, here we link it directly in the installed path, then export the path
export LD_LIBRARY_PATH=$PY_VENV_NVIDIA/cudnn/lib/:$PY_VENV_NVIDIA/cublas/lib/:$LD_LIBRARY_PATH

if [ ! -f "$PY_VENV_NVIDIA/cudnn/lib/libcudnn.so" ]; then
    cd $PY_VENV_NVIDIA/cudnn/lib
    ln -s ./libcudnn.so.* ./libcudnn.so
    cd -
fi
if [ ! -f "$PY_VENV_NVIDIA/cublas/lib/libcublas.so" ]; then
    cd $PY_VENV_NVIDIA/cublas/lib
    ln -s ./libcublas.so.* ./libcublas.so
    cd -
fi

MODEL_CKPT=${1:-"$CKPT_PATH/HelixFold3-240814.pdparams"}
INPUT_JSON=${2:-"data/demo_6zcy_smiles.json"}
OUTPUT_DIR=${3:-"./output"}

CUDA_VISIBLE_DEVICES=0 python /mnt/nvme/lijq/swbind/helixfold3-mmseqs2/inference.py \
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
    --infer_times 1 \
    --diff_batch_size 1 \
    --precision "fp32" \
    --search_tool "mmseqs"
    # hmmer | mmseqs

#CUDA_VISIBLE_DEVICES=0 python inference.py \
#    --skip_data_proc \
#    --sample ./output/demo_6zcy/final_features.pkl \
#    --init_model $CKPT_PATH/HelixFold3-240814.pdparams \
#    --max_template_date=2020-05-14 \
#    --input_json data/demo_6zcy.json \
#    --output_dir ./output \
#    --model_name allatom_demo \
#    --infer_times 1 \
#    --diff_batch_size 1 \
#    --precision "fp32"

#CUDA_VISIBLE_DEVICES=0 python inference.py \
#     --skip_inf \
#     --preset='reduced_dbs' \
#     --bfd_database_path "$MSA_PATH/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
#     --small_bfd_database_path "$MSA_PATH/small_bfd/bfd-first_non_consensus_sequences.fasta" \
#     --bfd_database_path "$MSA_PATH/small_bfd/bfd-first_non_consensus_sequences.fasta" \
#     --uniclust30_database_path "$MSA_PATH/uniclust30/uniclust30_2018_08/uniclust30_2018_08" \
#     --uniprot_database_path "$MSA_PATH/uniprot/uniprot.fasta" \
#     --pdb_seqres_database_path "$MSA_PATH/pdb_seqres/pdb_seqres.txt" \
#     --uniref90_database_path "$MSA_PATH/uniref90/uniref90.fasta" \
#     --mgnify_database_path "$MSA_PATH/mgnify/mgy_clusters_2018_12.fa" \
#     --template_mmcif_dir "$MSA_PATH/pdb_mmcif/mmcif_files" \
#     --obsolete_pdbs_path "$MSA_PATH/pdb_mmcif/obsolete.dat" \
#     --ccd_preprocessed_path "$MSA_PATH/ccd_preprocessed_etkdg.pkl" \
#     --rfam_database_path "$MSA_PATH/Rfam-14.9_rep_seq.fasta" \
#     --max_template_date=2020-05-14 \
#     --input_json data/demo_6zcy.json \
#     --output_dir ./output \
#     --model_name allatom_demo
