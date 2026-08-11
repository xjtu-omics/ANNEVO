# ANNEVO (v2.3.3)
**The training code has been open-sourced and is available in [`model_training`](model_training/).**

**Note: The ANNEVO model branches follow the NCBI Taxonomy classification. The name *Magnoliopsida* should not be interpreted as traditional dicotyledons. In NCBI Taxonomy (TaxID: 3398), Magnoliopsida represents flowering plants (angiosperms) and includes both monocots and dicots.**
## Recent Updates
1. Open-sourced the training code, available in [`model_training`](model_training/).
2. Added optional LZF compression for intermediate prediction H5 datasets. Compression is disabled by default and reduces prediction-file size without changing the final GFF output, but increases runtime due to the additional compression overhead.

## Quick version check
In the current version of ANNEVO, the BUSCO score for the **RefSeq human genome** annotation is shown as follows. In other performance evaluations of ANNEVO, this value can be used to check whether the latest version of ANNEVO was used.

| BUSCO_datasets | Value |
|----------------|------:|
| Mammalia_odb10 |  98.3 |
| Mammalia_odb12 |  98.2 |

Compared with the version described in the paper, some of the most noticeable changes in the latest version include:
1. The BUSCO score for human genome annotation increased from 95.7 to 98.3, using the same Mammalia_odb10 database as in the paper. 
2. On the same machine configuration described in the paper, the annotation time for the human genome decreased from 82 minutes to 19 minutes. 
3. The maximum predictable gene length in human increased from 621 kb to 1,212 kb (*CSMD3*). 
4. The minimum predictable gene length in human is 96 bp (*SLN*).

## Performance
**Evaluations were performed using the latest available version of each corresponding method as of June 1, 2026.** See [Notes](docs/notes.md) for details.
### Annotation accuracy
Annotation performance was evaluated on 12 model species across six clades, with two species selected from each clade. The average performance is shown below. The evaluation was performed using gffcompare ([Notes](docs/notes.md)). Detailed metrics for each species and the BUSCO databases used are available in [Performance](docs/performance.md).

| Method   | Exon recall | Exon precision | Locus recall | Locus precision | BUSCO |
|----------|------------:|---------------:|-------------:|----------------:|------:|
| ANNEVO   |        91.4 |           90.2 |         76.3 |            74.3 |  97.8 |
| Tiberius |        89.8 |           88.7 |         74.0 |            68.8 |  96.3 |
| Helixer  |        86.1 |           75.3 |         50.2 |            47.0 |  92.5 |

### Speed and GPU requirement
Runtime was evaluated on a single RTX 4090 GPU by calculating the average prediction time, in minutes, of the three methods across 12 model species. GPU memory requirements were measured on the human genome (GCF_000001405.40_GRCh38.p14). The detailed runtime for each species is provided in [Annotation time and resource usage](docs/performance.md).

| Method   | Time (single RTX4090, minutes) | GPU memory (GB, batch size=8) |
|----------|-------------------------------:|------------------------------:|
| ANNEVO   |                          12.18 |                           3.8 |
| Tiberius |                          43.59 |                          22.5 |
| Helixer  |                         286.61 |                           8.6 |

## Update history

See the complete [ANNEVO update history](docs/update_history.md).

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
# Available on 2026-06-05
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
conda install -c bioconda -c conda-forge bcbio-gff=0.7.1 h5py=3.14 torchmetrics=0.8.2 pandas=2.3.3 numpy=1.26.4 tqdm==4.67.1 numba=0.65.1
```

Check if CUDA is available:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

# Usage
ANNEVO provides both one-step execution and step-by-step execution.
## One-step Execution
```bash
python annotation.py -g path_to_genome -m path_to_model -l lineage -o path_to_gff --batch_size 32 -t 48 --show_log
```

## Parameter Description and Additional Parameters

| Parameter        | Description                                                                                                                                                                                                                                                                                                      |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-g`             | Genome file to be annotated.                                                                                                                                                                                                                                                                                     |
| `-m`             | Full path to the model file.                                                                                                                                                                                                                                                                                     |
| `-l`             | Lineage of the species to be predicted. This is used to determine the sequence segment length during inference. Supported lineages are `Mammalia`, `Insecta`, `Aves`, `Actinopteri`, `Magnoliopsida`, and `Fungi`.                                                                                               |
| `-o`             | Output GFF annotation file.                                                                                                                                                                                                                                                                                      |
| `-s`             | Chunk size used for each prediction run. This helps avoid repeatedly initializing PyTorch when processing highly fragmented genomes. This value affects peak memory usage in prediction step. The default value is `100`, representing 100 Mb. **If your available memory is limited, you can reduce this value. |
| `-t`             | Number of CPU cores used for decoding. This value affects decoding speed and peak memory usage in decoding step.                                                                                                                                                                                                 |
| `--show_log`     | Show the decoding progress.                                                                                                                                                                                                                                                                                      |
| `--overlap_pred` | Run overlapping-window prediction. For branches with relatively long genes, such as Mammalia and Actinopteri, we strongly recommend adding `--overlap_pred`, which can improve prediction performance to some extent. See [overlap_pred](docs/overlap_pred.md) for details.                                      |
| `--comp`         | Compress the intermediate prediction H5 datasets with LZF. Compression is disabled by default. This option reduces prediction-file size without changing the final GFF output, but increases runtime due to the additional compression overhead.                                                                 |

If your GPU environment has limited CPU resources, you can also use the step-by-step execution mode.
## Step-by-step Execution
Stage 1: Predict per-nucleotide probabilities from the genome sequence (recommended to be performed on environments with abundant **GPU** resources).  
Stage 2: Decode the model prediction into biologically valid gene structures (recommended to be performed on environments with abundant **CPU** resources).
```bash
# Nucleotide prediction
python prediction.py -g path_to_genome -m path_to_model -p path_to_prediction_h5 -l lineage --batch_size 32

# Gene structure decoding
python decoding.py -g path_to_genome -p path_to_prediction_h5 -o path_to_gff -t 48 --show_log
```
The `-p` parameter specifies the path for the output model prediction probability file. Add `--comp` to the prediction or one-step command to store this H5 file using LZF compression.
## Run demo data
The demo data located at './example'.
`Arabidopsis_chr4_genome.fna`: Genome sequence of chromosome 4 of Arabidopsis thaliana.
```bash
# One-step Execution
python annotation.py -g example/Arabidopsis_chr4_genome.fna -m saved_model/ANNEVO_Magnoliopsida.pt -l Magnoliopsida -o gff_result/Arabidopsis_chr4_annotation.gff -t 48 --show_log

# Step-by-step Execution
python prediction.py -g example/Arabidopsis_chr4_genome.fna -m saved_model/ANNEVO_Magnoliopsida.pt -p prediction_result/Arabidopsis_chr4/model_prediction.h5 -l Magnoliopsida
python decoding.py -g example/Arabidopsis_chr4_genome.fna -p prediction_result/Arabidopsis_chr4/model_prediction.h5 -o gff_result/Arabidopsis_chr4_annotation.gff -t 48 --show_log
```

# Contact
If you have any questions, please feel free to contact: pengyuzhang@stu.xjtu.edu.cn
