# ANNEVO (v2.3.1)
## Recent Updates
### v2.3.0
**New models have been updated for five major clades.** This update integrates all beneficial explorations made after the version described in the paper, including a new data processing workflow, new training strategies, minor architectural adjustments (position embedding), longer-context training, decoding algorithm and optimizations for practical resource usage and runtime. 

For plants, we continue to use the latest model from v2.2.3, because plants do not require longer-context training. Other optimizations had already been incorporated when that model was released, so we only further optimized the decoding algorithm. 

Compared with the version described in the paper, some of the most noticeable changes in the latest version include:
1. The BUSCO score for human genome annotation increased from 95.7 to 98.3, using the same Mammalia_odb10 database as in the paper. 
2. On the same machine configuration described in the paper, the annotation time for the human genome decreased from 82 minutes to 31 minutes. 
3. The maximum predictable gene length in human increased from 621 kb to 1,212 kb (*CSMD3*). 
4. The minimum predictable gene length in human is 96 bp (*SLN*).

### v2.3.1
v2.3.1 optimizes the chunking logic during the prediction stage, significantly improving memory resource management, especially for highly fragmented genome assemblies. For example, the RefSeq genome of *Chlamydotis macqueenii* (`GCF_000695195.1_ASM69519v1`) has a total genome size of 1.08 Gb and contains 59,693 sequences/contigs. The longest contig is only 399 kb, with an N50 of 45 kb. On this genome, the peak memory usage of ANNEVO during the prediction stage was reduced by approximately 80%.

For reference, under the default parameters, the peak memory usage of ANNEVO v2.3.1 during the prediction stage on the human genome (`GCF_000001405.40_GRCh38.p14`) is **34G**.

## Quick version check
In the current version of ANNEVO, the BUSCO score for the **RefSeq human genome** annotation is shown as follows. In other performance evaluations of ANNEVO, this value can be used to check whether the latest version of ANNEVO was used.

| BUSCO_datasets | Value |
|----------------|------:|
| Mammalia_odb10 |  98.3 |
| Mammalia_odb12 |  98.2 |


## Performance
**Evaluations were performed using the latest available version of each corresponding method as of May 18, 2026.** See [Notes](docs/notes.md) for details.
### Annotation accuracy
Annotation performance was evaluated on 12 model species across six clades, with two species selected from each clade. The average performance is shown below. The evaluation was performed using gffcompare ([Notes](docs/notes.md)). Detailed metrics for each species and the BUSCO databases used are available in [Performance](docs/performance.md).

| Method   | Exon recall | Exon precision | Locus recall | Locus precision | BUSCO |
|----------|------------:|---------------:|-------------:|----------------:|------:|
| ANNEVO   |        91.4 |           90.2 |         76.3 |            74.3 |  97.8 |
| Tiberius |        89.8 |           88.7 |         74.0 |            68.8 |  96.3 |
| Helixer  |        86.1 |           75.3 |         50.2 |            47.0 |  92.5 |

### Speed and GPU requirement
**Evaluations on human genome (GCF_000001405.40_GRCh38.p14).** See [Annotation time and resource usage](docs/performance.md) for details.

| Method   | Time (single RTX4090, minutes) | Time (CPU only, minutes) | GPU memory (GB, batch size=8) |
|----------|-------------------------------:|-------------------------:|------------------------------:|
| ANNEVO   |                             52 |                 7.4 * 60 |                           3.8 |
| Tiberius |                            141 |                 8.8 * 60 |                          22.5 |
| Helixer  |                            956 |                  23 * 60 |                           8.6 |

As a quick comparison on a small genome, using the same single RTX 4090, ANNEVO took 1.9 minutes on Arabidopsis thaliana, compared with 5.7 minutes for Tiberius and 13.9 minutes for Helixer.

## Update history
#### 2026-05 (v2.3.1): Memory usage optimization, especially for fragmented genome assemblies.
#### 2026-05 (v2.3.0): Covering updates to the model, data, training strategy, and engineering implementation, with substantially improved performance and speed.
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
The `-p` parameter specifies the path for the output model prediction probability file.
## Run demo data
The demo data located at './example'.
`Arabidopsis_chr4_genome.fna`: Genome sequence of chromosome 4 of Arabidopsis thaliana.
```bash
# One-step Execution
python annotation.py -g example/Arabidopsis_chr4_genome.fna -m saved_model/ANNEVO_Magnoliopsida_seq.pt -l Magnoliopsida -o gff_result/Arabidopsis_chr4_annotation.gff -t 48 --show_log

# Step-by-step Execution
python prediction.py -g example/Arabidopsis_chr4_genome.fna -m saved_model/ANNEVO_Magnoliopsida_seq.pt -p prediction_result/Arabidopsis_chr4/model_prediction.h5 -l Magnoliopsida
python decoding.py -g example/Arabidopsis_chr4_genome.fna -p prediction_result/Arabidopsis_chr4/model_prediction.h5 -o gff_result/Arabidopsis_chr4_annotation.gff -t 48 --show_log
```

# Contact
If you have any questions, please feel free to contact: pengyuzhang@stu.xjtu.edu.cn
