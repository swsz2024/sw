# Installation
HelixFold3 depends on [PaddlePaddle](https://github.com/paddlepaddle/paddle). Python dependencies available through `pip` 
is provided in `dev-requirements.txt`. `jackhmmer` is needed to produce multiple sequence alignments. The download scripts require `aria2c`(optional). 

The hardware requirements. [TBD] 

## Obtaining HelixFold3 Source Code

Download the Helixfold3 repository from gitlab into `$HELIXFOLD3_SOURCE_DIR`:

From IP of NSCCWX, get from local ip `192.167.253.229:25681`
```sh
git clone ssh://git@192.167.253.229:25682/ai4s/swbind/helixfold3.git $HELIXFOLD3_SOURCE_DIR
```
From IP of LIHU-FUTURE-CITY, get from pubilc ip `58.214.1.186:25681`
```sh
git clone ssh://git@58.214.1.186:25682/ai4s/swbind/helixfold3.git $HELIXFOLD3_SOURCE_DIR
```

## Installation with Dockerfile(deployment-only)
We recommend to use docker to run the helixfold3 inference.
The instructions provided below describe how to:

1.  Install Docker.
1.  Build the HelixFold 3 Docker container image.
1.  Run your first prediction

### Installing Docker

First to install docker. There are many guides on the Internet.

1. Installing Docker on Host
1. Installing GPU Support: NVIDIA Drivers and NVIDIA Support for Docker


### Building the HelixFold 3 Docker container image.
Then, build the Docker container. This builds a container with all the right
python dependencies:

```sh
cd $HELIXFOLD3_SOURCE_DIR
docker build -t helixfold3 -f docker/Dockerfile .
```

You can now run HelixFold 3!

## Installation with Local Environment(dev-only)
You can also run helixfold3 with local environment. Now we use devices of
`LIHU-FUTURE-CITY` to develop. We can use the instructions provided below in the
dev-container to install HelixFold 3:

1.  Setup system environment
1.  Setup python environment.
1.  Install Hmmer via source files
1.  Run your first prediction

### Setup System Environment
```bash
# Install gcc g++ cmake
apt update --quiet
apt install --yes --quiet wget gcc g++ cmake

# Install cuda
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
# Deselect the driver during installation
sh cuda_12.0.0_525.60.13_linux.run
rm cuda_12.0.0_525.60.13_linux.Run
# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
```

### Setup Python Environment
```bash
# Install py env
conda create -n helixfold python=3.9
conda activate helixfold
cd $HELIXFOLD3_SOURCE_DIR
python3 -m venv .venv

# Activate the py env 
source .venv/bin/activate

# Install requirements
pip install -r dev-requirements.txt
pip install --no-deps .
```

### Install HMMER Via Source Files
```bash
# Download hmmer source code 
mkdir -p $HELIXFOLD3_SOURCE_DIR/../hmmer/src
cd $HELIXFOLD3_SOURCE_DIR/../hmmer/src 
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz

# Install hmmer
tar -zxf hmmer-3.4.tar.gz
cd hmmer-3.4
./configure --prefix $HELIXFOLD3_SOURCE_DIR/../hmmer
make -j8  && make install
```
You can now run HelixFold 3!

## Run Your First Prediction

### Obtaining Genetic Databases

HelixFold 3 needs multiple genetic (sequence) protein and RNA databases to run:

*   [BFD small](https://bfd.mmseqs.com/)
*   [MGnify](https://www.ebi.ac.uk/metagenomics/)
*   [PDB](https://www.rcsb.org/) (structures in the mmCIF format)
*   [PDB seqres](https://www.rcsb.org/)
*   [UniProt](https://www.uniprot.org/uniprot/)
*   [UniRef90](https://www.uniprot.org/help/uniref)
*   [NT](https://www.ncbi.nlm.nih.gov/nucleotide/)
*   [RFam](https://rfam.org/)
*   [RNACentral](https://rnacentral.org/)

The databases are pre-downloaded.
In `LIHU-FUTURE-CITY`, the data is in the `SWBind_share` storage bucket.
```
<SWBind_share>
└-- data
    |-- ckpt
    |   |-- af3
    |   `-- helixfold
    |       |-- HelixFold3-240814.pdparams
    |       |-- LICENSE
    |       |-- helixfold_aa_model_e26_221_240920.pdparams
    |       `-- step_120000.pdparams
    `-- msa_datasets
        |-- af3
        `-- helixfold
            |-- Rfam-14.9_rep_seq.fasta
            |-- bfd
            |   `-- bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
            |-- ccd_preprocessed_etkdg.pkl
            |-- chembl_ccd.csv
            |-- mgnify
            |   `-- mgy_clusters_2018_12.fa
            |-- pdb_seqres
            |   `-- pdb_seqres.txt
            |-- small_bfd
            |   `-- bfd-first_non_consensus_sequences.fasta
            |-- uniclust30
            |   `-- uniclust30_2018_08
            |       |-- uniclust30_2018_08.cs219
            |       |-- uniclust30_2018_08.cs219.sizes
            |       |-- uniclust30_2018_08_a3m.ffdata
            |       |-- uniclust30_2018_08_a3m.ffindex
            |       |-- uniclust30_2018_08_a3m_db -> uniclust30_2018_08_a3m.ffdata
            |       |-- uniclust30_2018_08_a3m_db.index
            |       |-- uniclust30_2018_08_cs219.ffdata
            |       |-- uniclust30_2018_08_cs219.ffindex
            |       |-- uniclust30_2018_08_hhm.ffdata
            |       |-- uniclust30_2018_08_hhm.ffindex
            |       |-- uniclust30_2018_08_hhm_db -> uniclust30_2018_08_hhm.ffdata
            |       |-- uniclust30_2018_08_hhm_db.index
            |       `-- uniclust30_2018_08_md5sum
            |-- uniprot
            |   `-- uniprot.fasta
            |-- uniref90
            |    `-- uniref90.fasta
            `-- pdb_mmcif
                |-- obsolete.dat
                `-- mmcif_files
                    |-- 6zcy
                    |   |-- ....
                    |   `-- ....  
                     `-- ....
```
The related MSA databases used by helixfold3 are all in `<SWBind_share>/data/msa_datasets/helixfold`.
Commonly, the `<SWBind_share>/data` will be mounted into the container as `/mnt/share/`
The ckpts of helixfold3 are in `<SWBind_share>/data/ckpt/helixfold/`.

In `NSCC-YANCHENG`, the data is in `/usb/zmj/msa_datasets/helixfold` and `/usb/zmj/ckpt/helixfold`
separately.

### Run The Prediction 

To run inference on a sequence or multiple sequences using HelixFold3's pretrained parameters, run e.g.:
* Inference on single GPU (change the settings in script BEFORE you run it)
```
sh run_infer.sh
```

The script is as follows,
```bash
#!/bin/bash

MSA_PATH="/mnt/data/msa_datasets/origin"
CKPT_PATH="/mnt/data/ckpt/helixfold"
PY_VENV=./.venv
PY_VENV_NVIDIA=$PY_VENV/lib/python3.9/site-packages/nvidia
HMMER_PATH=../hmmer/bin

export PATH=$HMMER_PATH:$PATH
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

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --preset='reduced_dbs' \
    --bfd_database_path "$MSA_PATH/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
    --small_bfd_database_path "$MSA_PATH/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --bfd_database_path "$MSA_PATH/small_bfd/bfd-first_non_consensus_sequences.fasta" \
    --uniclust30_database_path "$MSA_PATH/uniclust30/uniclust30_2018_08/uniclust30_2018_08" \
    --uniprot_database_path "$MSA_PATH/uniprot/uniprot.fasta" \
    --pdb_seqres_database_path "$MSA_PATH/pdb_seqres/pdb_seqres.txt" \
    --uniref90_database_path "$MSA_PATH/uniref90/uniref90.fasta" \
    --mgnify_database_path "$MSA_PATH/mgnify/mgy_clusters_2018_12.fa" \
    --template_mmcif_dir "$MSA_PATH/pdb_mmcif/mmcif_files" \
    --obsolete_pdbs_path "$MSA_PATH/pdb_mmcif/obsolete.dat" \
    --ccd_preprocessed_path "$MSA_PATH/ccd_preprocessed_etkdg.pkl" \
    --rfam_database_path "$MSA_PATH/Rfam-14.9_rep_seq.fasta" \
    --init_model $CKPT_PATH/HelixFold3-240814.pdparams \
    --max_template_date=2020-05-14 \
    --input_json data/demo_6zcy.json \
    --output_dir ./output \
    --model_name allatom_demo \
    --infer_times 1 \
    --diff_batch_size 1 \
    --precision "fp32"
```

The descriptions of the above script are as follows:
* Replace `MSA_PATH` with your MSA databases path.
* Replace `CKPT_PATH` with your model checkpoints path.
* Replace `PY_VENV` with your installed python virtualenv path.
* Replace `HMMER_PATH` with your installed Hmmer software path.
* `--preset` - Set `'reduced_dbs'` to use small bfd.
* `--*_database_path` - Path to datasets you have downloaded.
* `--input_json` - Input data in the form of JSON. Input pattern in `./data/demo_*.json` for your reference.
* `--output_dir` - Model output path. The output will be in a folder named the same as your `--input_json` under this path.
* `--model_name` - Model name in `./helixfold/model/config.py`. Different model names specify different configurations. Mirro modification to configuration can be specified in `CONFIG_DIFFS` in the `config.py` without change to the full configuration in `CONFIG_ALLATOM`.
* `--infer_time` - The number of inferences executed by model for single input. In each inference, the model will infer `5` times (`diff_batch_size`) for the same input by default. This hyperparameter can be changed by `model.head.diffusion_module.test_diff_batch_size` within `./helixfold/model/config.py`
* `--precision` - Either `bf16` or `fp32`. Please check if your machine can support `bf16` or not beforing changing it. For example, `bf16` is supported by A100 and H100 or higher version while V100 only supports `fp32`.


### Use Data From `.pkl`

Preprocessed data in terms of `.pkl` can also be used in inference rather than processing data from 
scratch. All preprocessed data is almost the same as previously provided expect some additional features. 
Those additional features are irrelevant to diffusion results (i.e., final atom coordinates) and they are 
required by confidence head for metrics only. Both version have the same token numbers and atom numbers, but 
the new version may have deeper MSA due to MSA cropping in previous implementation. Therefore, previous 
preprocessed data is no longer supported by this version.

Add `--skip_data_proc` in args to avoid processing data. Please also specify the sample `.pkl` path. You may crop MSA if they occupy too much memory.

```bash

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --skip_data_proc \
    --sample ./output/demo_6zcy/final_features.pkl \
    --init_model $CKPT_PATH/HelixFold3-240814.pdparams \
    --max_template_date=2020-05-14 \
    --input_json data/demo_6zcy.json \
    --output_dir ./output \
    --model_name allatom_demo \
    --infer_times 1 \
    --diff_batch_size 1 \
    --precision "fp32"

```

### Run With the Container 

The script to run helixfold3 with the built container.

```bash
#paths in container, fixed.
CONTAINER_MSA_PATH="/msa/"
CONTAINER_CKPT_PATH="/ckpt/"
CONTAINER_INPUT="/input"
CONTAINER_OUTPUT="/output"

# paths in the host, specify according to the fact
HOST_MSA_PATH="/usb/zmj/af3"
HOST_CKPT_PATH="/usb/zmj/af3/params"
HOST_INPUT="./input"
HOST_OUTPUT="./output"

# GPU CONFIGS
GPU_CONFIG="device=0"

docker run -it \
    --volume $HOST_MSA_PATH/:$CONTAINER_MSA_PATH \
    --volume $HOST_CKPT_PATH/:$CONTAINER_CKPT_PATH \
    --volume $HOST_OUTPUT:$CONTAINER_OUTPUT \
    --volume $HOST_INPUT:$CONTAINER_INPUT \
    --gpus $GPU_CONFIG \
    helixfold3 \
    python inference.py \
        --preset='reduced_dbs' \
        --bfd_database_path "$CONTAINER_MSA_PATH/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
        --small_bfd_database_path "$CONTAINER_MSA_PATH/small_bfd/bfd-first_non_consensus_sequences.fasta" \
        --bfd_database_path "$CONTAINER_MSA_PATH/small_bfd/bfd-first_non_consensus_sequences.fasta" \
        --uniclust30_database_path "$CONTAINER_MSA_PATH/uniclust30/uniclust30_2018_08/uniclust30_2018_08" \
        --uniprot_database_path "$CONTAINER_MSA_PATH/uniprot/uniprot.fasta" \
        --pdb_seqres_database_path "$CONTAINER_MSA_PATH/pdb_seqres/pdb_seqres.txt" \
        --uniref90_database_path "$CONTAINER_MSA_PATH/uniref90/uniref90.fasta" \
        --mgnify_database_path "$CONTAINER_MSA_PATH/mgnify/mgy_clusters_2018_12.fa" \
        --template_mmcif_dir "$CONTAINER_MSA_PATH/pdb_mmcif/mmcif_files" \
        --obsolete_pdbs_path "$CONTAINER_MSA_PATH/pdb_mmcif/obsolete.dat" \
        --ccd_preprocessed_path "$CONTAINER_MSA_PATH/ccd_preprocessed_etkdg.pkl" \
        --rfam_database_path "$CONTAINER_MSA_PATH/Rfam-14.9_rep_seq.fasta" \
        --init_model $CONTAINER_CKPT_PATH/HelixFold3-240814.pdparams \
        --max_template_date=2020-05-14 \
        --input_json $CONTAINER_INPUT/demo_6zcy.json \
        --output_dir $CONTAINER_OUTPUT \
        --model_name allatom_demo \
        --infer_times 1 \
        --diff_batch_size 1 \
        --precision "fp32"
```
Also, we can run with `--skip_data_proc`.

```bash
docker run -it \
    --volume $HOST_MSA_PATH/:$CONTAINER_MSA_PATH \
    --volume $HOST_CKPT_PATH/:$CONTAINER_CKPT_PATH \
    --volume $HOST_OUTPUT:$CONTAINER_OUTPUT \
    --volume $HOST_INPUT:$CONTAINER_INPUT \
    --gpus $GPU_CONFIG \
    helixfold3 \
    python inference.py \
        --skip_data_proc
        --sample $CONTAINER_OUTPUT/demo_6zcy/final_features.pkl
        --init_model $CONTAINER_CKPT_PATH/HelixFold3-240814.pdparams \
        --max_template_date=2020-05-14 \
        --input_json $CONTAINER_INPUT/demo_6zcy.json \
        --output_dir $CONTAINER_OUTPUT \
        --model_name allatom_demo \
        --infer_times 1 \
        --diff_batch_size 1 \
        --precision "fp32"

```
