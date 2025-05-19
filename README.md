# Update (Current version: 2.0)
As stated in our manuscript, ANNEVO has not undergone systematic hyperparameter tuning or optimization, and maintains a deliberately lightweight model architecture. Therefore, its current performance is far from optimal, yet it still achieves state-of-the-art results. To further explore the potential of ANNEVO in genome annotation, we retained its core innovations (Hi-C inspired modeling and evolutionary Mixture-of-Experts network) while implementing several additional optimizations:  
(1) A more efficient and rational model structure.  
We replaced the transformer encoder with linear layers within each expert network, reducing resource requirements and increasing computational speed.
For restoring nucleotide-level resolution, we adopted a progressive transposed convolution approach instead of a direct reshape operation, thereby minimizing information loss during reconstruction.  
(2) A more rigorous decoding algorithm.  
We continue to predict the primary class of each nucleotide and map it to multiple states. Additionally, we developed a custom CDS state framework that completely eliminates premature stop codons within coding frames, even for codons spanning introns.  

**All training, validation, and test datasets used in the new version are exactly the same as those used in the previous version described in the manuscript, ensuring that the improvements are directly comparable.** The previous version can be found in the v1 directory (https://github.com/xjtu-omics/ANNEVO/tree/main/v1).
## Improvements
### Performance 
Performance evaluations have been conducted on 12 model species (***Fungi, Plant, Mammalia, Vertebrate_other and Invertebrate***) as described in the manuscript, with results showing **significant performance improvements**.   
**ANNEVO has now significantly outperform BRAKER3**. Detailed metrics for certain species and descriptions can be found in the *performance_evaluation* directory (https://github.com/xjtu-omics/ANNEVO/tree/main/performance_evaluation/Performance.md).

| Model     | NT(CDS)-F1 | gene-F1 | BUSCO |
|:----------|:----------:|:-------:|------:|
| Augustus  |    72.4    |  49.7   |  73.5 |
| Helixer   |    87.2    |  70.8   |  90.3 |
| BRAKER3   |    84.6    |  76.4   |  93.9 |
| ANNEVO v1 |    88.5    |  75.3   |  93.9 |
| ANNEVO v2 |    91.7    |  80.5   |  96.6 |

### Computational time
Computational time advantage will be more significant in plants, because ANNEVO only needs to decode potential gene regions, and the length of gene regions in plants is much smaller than that in mammals (gene length is shorter). BRAKER3 need to decode all regions. Note: When using the one-step command, ANNEVO no longer needs to store intermediate prediction files, which significantly reduces runtime.

| Model                                | Sus scrofa (2.36G) | Zea mays (2.06G) | A. thaliana (0.12G) | 
|:-------------------------------------|:------------------:|:----------------:|--------------------:|
| BRAKER3                              |       47.6h        |      102.7h      |              972min | 
| ANNEVO v1 (Step-by-step Execution)   |       2.94h        |      1.46h       |              5.8min | 
| ANNEVO v2 (Step-by-step Execution)   |       1.28h        |      0.79h       |              3.5min | 
| ANNEVO v2 (One-step Execution)       |       1.13h        |      0.64h       |              3.4min |

# ANNEVO
ANNEVO is a deep learning-based ab initio gene annotation method for understanding genome function. ANNEVO is capable of modeling distal sequence information and joint evolutionary relationships across diverse species directly from genomes.  

ANNEVO is designed to model various sub-lineages at high taxonomic levels while simultaneously accounting for distal interactions within the genome. It comprises three main components: (a) Context Extension: each nucleotide is provided with sufficient contextual information and regions are masked to reduce their contribution due to likely errors. (b) Neural Network: Modeling of both long-range interactions within sequences and multiple sub-lineages using a broad range of species enables end-to-end predictions of category, phase, and state for each nucleotide. (c) Gene Structure Decoding: Connects prediction result for individual segments to identify potential gene structures.
![GitHub Image](https://raw.githubusercontent.com/xjtu-omics/ANNEVO/main/img/Fig1.png)
## License
ANNEVO is free for non-commercial use by academic, government, and non-profit/not-for-profit institutions. A commercial version of the software is available and licensed through Xi'an Jiaotong University. For more information, please contact with Pengyu Zhang (pengyuzhang@stu.xjtu.edu.cn) or Kai Ye (kaiye@xjtu.edu.cn).  

## Installation
We recommend using the conda virtual environment to install ANNEVO (Platform: Linux).
```bash
# Get the source code
git clone https://github.com/xjtu-omics/ANNEVO.git
cd ANNEVO

# Create a conda environment for ANNEVO
conda create -n ANNEVO python=3.6

# Activate conda environment
conda activate ANNEVO

# To use GPU acceleration properly, we recommend installing PyTorch using the official installation 
# commands provided by PyTorch (https://pytorch.org/get-started/previous-versions/). 
# Select the appropriate command based on your CUDA version to install PyTorch version 1.10. 
# Or directly use `pytorch-cuda` to automatically install the appropriate `cudatoolkit`. 
# For example, if the CUDA version is not lower than 11.8, you can use the following command:
conda install pytorch=1.10 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install .
conda install -c bioconda seqkit=2.9.0
```

Check if CUDA is available:
```bash
import torch
print(torch.cuda.is_available())
```

## Usage
### One-step Execution
```bash
python annotation.py --genome path_to_genome --lineage selected_lineage --output path_to_gff --threads 48
```
**Optional lineage: Fungi, Embryophyta, Invertebrate, Vertebrate_other, Mammalia.**  
We strongly recommend utilizing more CPU cores by adjusting threads when sufficient computational resources are available, as this will significantly accelerate the computation. If your GPU environment has limited CPU resources, you can also use the step-by-step execution mode.  
Note: ANNEVO automatically supports use in a multi-GPU environment. If GPU resources are insufficient, you can adjust it by `--batch-size`. For example, adding the parameter `--batch-size 8` only requires about 4G GPU memory.

### Step-by-step Execution
Typically, deep learning is conducted in environments equipped with GPU resources, where CPU resources are often limited. However, decoding gene structures usually requires substantial CPU resources. To address this, we provide a segmented execution approach, allowing users to flexibly switch between computational nodes/environments with different resources.  
Stage 1: Predicting three types of information for each nucleotide (recommended to be performed on environments with abundant GPU resources).  
Stage 2: Decoding the three types of information into biologically valid gene structures (recommended to be performed on environments with abundant CPU resources).
```bash
# Nucleotide prediction
python prediction.py --genome path_to_genome --lineage selected_lineage --model_prediction_path path_to_save_predction

# Gene structure decoding
python decoding.py --genome path_to_genome --model_prediction_path path_to_save_predction --output path_to_gff --threads 48 
```
### Run demo data
The demo data located at './example'.  
`Arabidopsis_chr4_genome.fna`: Genome sequence of chromosome 4 of Arabidopsis thaliana.  
`Arabidopsis_chr4_annotation.gff`: RefSeq annotation of chromosome 4 of Arabidopsis thaliana.
```bash
# One-step Execution
python annotation.py --genome example/Arabidopsis_chr4_genome.fna --lineage Embryophyta --output gff_result/Arabidopsis_chr4_annotation.gff --threads 48

# Step-by-step Execution
python prediction.py --genome example/Arabidopsis_chr4_genome.fna --lineage Embryophyta --model_prediction_path prediction_result/Arabidopsis_chr4
python decoding.py --genome example/Arabidopsis_chr4_genome.fna --model_prediction_path prediction_result/Arabidopsis_chr4 --output gff_result/Arabidopsis_chr4_annotation.gff --threads 48
```

# Contact
If you have any questions, please feel free to contact: pengyuzhang@stu.xjtu.edu.cn