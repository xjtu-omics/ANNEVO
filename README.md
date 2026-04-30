# ANNEVO (v2.2.3)
## Recent Updates
1. **Released a new model for the `Magnoliopsida` clade (flowering plants, Tax ID: 3398), covering both `monocots` and `dicots`.** Preliminary evaluation (using gffcompare) on two representative species are provided below. See [boundary_aware_model](docs/boundary_aware_model.md) for details and usage instructions.
2. **Optimized the parallel decoding logic for large genomes and multithreaded settings. This improvement applies to all model.** In our evaluation, decoding time on the human genome was reduced from ~2800 s to ~1900 s (~30%). 
3. **Improved the logic for applying `min_intron_length`**, so that for most gene segments it no longer introduces extra decoding time.

**Evaluations were performed using the latest available version of each corresponding method as of April 29, 2026.**

| Species | Model | Exon-recall | Exon-precision | Locus-recall | Locus-precision |
|---|---|---:|---:|---:|---:|
| A.thaliana | Helixer: land_plant_v0.3_a_0080.h5 | 89.5 | 88.1 | 75.1 | 75.6 |
| A.thaliana | Tiberius: angiosperms | 89.6 | 94 | 80.5 | 88.2 |
| A.thaliana | ANNEVO: Magnoliopsida | 89.6 | 94.2 | 81.3 | 89.3 |
|  |  |  |  |  |  |
| O.sativa | Helixer: land_plant_v0.3_a_0080.h5 | 88.9 | 71.4 | 68.7 | 53.1 |
| O.sativa | Tiberius: angiosperms | 87.4 | 88.2 | 74.6 | 71.3 |
| O.sativa | ANNEVO: Magnoliopsida | 90.5 | 89.6 | 80.1 | 78.5 |


## Update history
#### 2026-04 (v2.2.3): Added a new plant model and improved over 30% decoding speed.
#### 2026-03 (v2.2.2): Optimized the search logic for candidate intervals during decoding.
#### 2026-01 (v2.2.1): Released two new models for Insecta and Mammalia, trained with the new data processing and training pipeline.
#### 2025-10 (v2.2): Memory usage optimization.  
#### 2025-07 (v2.1): New model architecture and training procedure.
#### 2025-01 (v1.0): Ab initio gene annotation with ANNEVO.
## Overview
ANNEVO is a deep learning-based ab initio gene annotation method for understanding genome function. ANNEVO is capable of modeling distal sequence information and joint evolutionary relationships across diverse species directly from genomes.  

![GitHub Image](https://raw.githubusercontent.com/xjtu-omics/ANNEVO/main/img/Fig1.png)
## License
ANNEVO is distributed under the ANNEVO Non-Commercial License. It is free for academic and non-profit research use.  
Commercial use requires a separate license. For commercial use or licensing inquiries, please contact: Pengyu Zhang (pengyuzhang@stu.xjtu.edu.cn) or Kai Ye (kaiye@xjtu.edu.cn).  
Note: ANNEVO is not licensed under the GNU GPL or any OSI-approved open source license.
It is distributed under the ANNEVO Non-Commercial License, which restricts commercial use.

# Installation
Note: We found that, in some specific cases, installation failures were mainly caused by version changes in the dependencies of certain packages, which made it impossible to satisfy all version requirements simultaneously. To address this, we adjusted the installation sources for some dependencies so that the environment can now be installed directly from the YAML file. We will check once per month whether the YAML file remains directly installable, to ensure a smooth and convenient installation experience for users.  

We recommend using the conda virtual environment to install ANNEVO (Platform: Linux).
```bash
# Get the source code
git clone https://github.com/xjtu-omics/ANNEVO.git
cd ANNEVO
```
If your CUDA version is higher than 12.1, you can directly install the environment using:
```
# Available on 2026-04-17 
conda env create -f ANNEVO.yml -n your_env_name
```
Alternatively, you can follow the steps below to install the environment manually.
This is especially recommended for users with lower CUDA versions, as you may need to manually adjust the PyTorch version and installation source.
```
# Create a conda environment for ANNEVO
conda create -n ANNEVO python=3.10

# Activate conda environment
conda activate ANNEVO

# To use GPU acceleration properly, we recommend installing PyTorch using the 
# official installation commands provided by PyTorch (https://pytorch.org/get-started/previous-versions/). 
# A sample installation command is shown below:
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
conda install -c bioconda -c conda-forge bcbio-gff=0.7.1 h5py=3.14 torchmetrics=0.8.2 pandas=2.3.3 numpy=1.26.4 tqdm==4.67.1
```

Check if CUDA is available:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

# Usage
Note: Now you can use `--show_log` to view the decoding progress.
## One-step Execution
```bash
python annotation.py --genome path_to_genome --model_path path_to_model --output path_to_gff --threads 48
```
We strongly recommend utilizing more CPU cores by adjusting threads when sufficient computational resources are available, as this will significantly accelerate the computation. If your GPU environment has limited CPU resources, you can also use the step-by-step execution mode.  
Note: ANNEVO automatically supports use in a multi-GPU environment. If GPU resources are insufficient, you can adjust it by `--batch_size`. For example, adding the parameter `--batch_size 8` only requires <2G GPU memory.

## Step-by-step Execution
Typically, deep learning is conducted in environments equipped with GPU resources, where CPU resources are often limited. However, decoding gene structures usually requires substantial CPU resources. To address this, we provide a segmented execution approach, allowing users to flexibly switch between computational nodes/environments with different resources.  
Stage 1: Predicting three types of information for each nucleotide (recommended to be performed on environments with abundant GPU resources).  
Stage 2: Decoding the three types of information into biologically valid gene structures (recommended to be performed on environments with abundant CPU resources).
```bash
# Nucleotide prediction
python prediction.py --genome path_to_genome --model_path path_to_model --model_prediction_path path_to_save_predction

# Gene structure decoding
python decoding.py --genome path_to_genome --model_prediction_path path_to_save_predction --output path_to_gff --threads 48 
```
## Run demo data
The demo data located at './example'.  
`Arabidopsis_chr4_genome.fna`: Genome sequence of chromosome 4 of Arabidopsis thaliana.  
`Arabidopsis_chr4_annotation.gff`: RefSeq annotation of chromosome 4 of Arabidopsis thaliana.
```bash
# One-step Execution
python annotation.py --genome example/Arabidopsis_chr4_genome.fna --model_path ANNEVO_model/ANNEVO_Embryophyta.pt --output gff_result/Arabidopsis_chr4_annotation.gff --threads 48

# Step-by-step Execution
python prediction.py --genome example/Arabidopsis_chr4_genome.fna --model_path ANNEVO_model/ANNEVO_Embryophyta.pt --model_prediction_path prediction_result/Arabidopsis_chr4/model_prediction.h5
python decoding.py --genome example/Arabidopsis_chr4_genome.fna --model_prediction_path prediction_result/Arabidopsis_chr4/model_prediction.h5 --output gff_result/Arabidopsis_chr4_annotation.gff --threads 48
```
# Re-train or Fine-tune ANNEVO
See [Re-train_and_fine-tune](docs/Re-train_and_fine-tune.md) for details and usage instructions.

# Contact
If you have any questions, please feel free to contact: pengyuzhang@stu.xjtu.edu.cn