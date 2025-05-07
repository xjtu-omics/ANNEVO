# Update
As stated in our manuscript, ANNEVO has not undergone systematic hyperparameter tuning or optimization, and maintains a deliberately lightweight model architecture. Therefore, its current performance is far from optimal, yet it still achieves state-of-the-art results. To further explore the potential of ANNEVO in genome annotation, we retained its core innovations (Hi-C inspired modeling and evolutionary Mixture-of-Experts network) while implementing several additional optimizations:  
(1) A more efficient and rational model structure.  
We replaced the transformer encoder with linear layers within each expert network, reducing resource requirements and increasing computational speed.
For restoring nucleotide-level resolution, we adopted a progressive transposed convolution approach instead of a direct reshape operation, thereby minimizing information loss during reconstruction.  
(2) A more rigorous decoding algorithm.  
We continue to predict the primary class of each nucleotide and map it to multiple states. Additionally, we developed a custom CDS state framework that completely eliminates premature stop codons within coding frames, even for codons spanning introns.  

The new version can be found in the v2.0 directory (https://github.com/xjtu-omics/ANNEVO/tree/main/v2.0). The new version is currently available for ***Mammalia, Embryophyta and Fungi***, and preliminary evaluations have been conducted on several model organisms, with results showing **significant performance improvements. ANNEVO has now surpassed BRAKER3 in all available lineages**. 
### Average performance of 6 model species (Fungi, Plant and Mammalia) as described in the manuscript
| Model   | NT(CDS)-F1 | gene-F1 | BUSCO |
|:--------|:----------:|:-------:|------:|
| BRAKER3 |    83.5    |  72.8   |  92.1 |
| ANNEVO  |    91.2    |  78.3   |  97.0 |

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
Typically, deep learning is conducted in environments equipped with GPU resources, where CPU resources are often limited. However, decoding gene structures usually requires substantial CPU resources. To address this, we provide a segmented execution approach, allowing users to flexibly switch between computational nodes/environments with different resources. ANNEVO consists of two stages:  
Stage 1: Predicting three types of information for each nucleotide (recommended to be performed on environments with abundant GPU resources).  
Stage 2: Decoding the three types of information into biologically valid gene structures (recommended to be performed on environments with abundant CPU resources).  
User can use the following two instructions in the current directory to complete gene annotation:

### Nucleotide prediction
The required parameters for the first stage include the path to the genome, the specified lineage (which determines the lineage-specific model parameters to be used; Options: Fungi, Embryophyta, Mammalia, Vertebrate_other, Invertebrate), and the path to save the prediction results. The command for nucleotide prediction is as follows:
```bash
python -m ANNEVO.prediction --genome path_to_genome --lineage selected_lineage --model_prediction_path path_to_save_predction
```
Regarding the balance of computing resources and computing time, users can further adjust the `chunk_num`, `batch_size` and `num_workers` parameters.

### Gene structure decoding
The required parameters for the second stage include the path to the genome, the path to the model prediction results, and the output annotation file. The command for gene structure decoding is as follows:
```bash
python -m ANNEVO.decoding --genome path_to_genome --model_prediction_path path_to_save_predction --output path_to_gff --lineage selected_lineage --threads 48 --min_intron_length 30
```
We strongly recommend utilizing more CPU cores by adjusting `threads` when sufficient computational resources are available, as this will significantly accelerate the computation.

### Run demo data
The demo data located at './example'.  
`Arabidopsis_chr4_genome.fna`: Genome sequence of chromosome 4 of Arabidopsis thaliana.  
`Arabidopsis_chr4_annotation.gff`: RefSeq annotation of chromosome 4 of Arabidopsis thaliana.
```bash
python -m ANNEVO.prediction --genome example/Arabidopsis_chr4_genome.fna --lineage Embryophyta --model_prediction_path prediction_result/Arabidopsis_chr4
python -m ANNEVO.decoding --genome example/Arabidopsis_chr4_genome.fna --model_prediction_path prediction_result/Arabidopsis_chr4 --output gff_result/Arabidopsis_chr4_annotation.gff --threads 48 --min_intron_length 30 --lineage Embryophyta
```
You will see the prediction results in `gff_result/Arabidopsis_chr4_annotation.gff`.

## Retrain ANNEVO (Optional)
ANNEVO supports the retraining of specific lineages using additional genomic data to further optimize performance. Using the demonstration data as an example, the first step is to preprocess the data based on the genome and annotation (we strongly recommend adjusting `cpu_num` to utilize more CPU cores when sufficient computational resources are available):
```bash
# Filter GFF file to remove entries with duplicate gene IDs and their associated sub-features.
python -m ANNEVO.src.filter_wrong_record --input_file example/Arabidopsis_chr4_annotation.gff --output_file example/filterred_Arabidopsis_chr4_annotation.gff
python -m ANNEVO.data_processing --genome example/Arabidopsis_chr4_genome.fna --annotation example/filterred_Arabidopsis_chr4_annotation.gff --output_file processed_data/Arabidopsis_chr4.h5
```
The training process typically requires the genomes of multiple species. Therefore, ANNEVO provides a `species_list.txt` to index the training species and validation species.
```bash
python -m ANNEVO.train --train_list example/train_species_list.txt --val_list example/val_species_list.txt --model_save_path ANNEVO/saved_model/ANNEVO_test.pt --h5_path processed_data/
```
Note: ANNEVO supports personalized training, such as local pattern extraction depth and sub-lineage count. Please refer to the detailed instructions by running `python -m ANNEVO.train -h`.
## Contact
If you have any questions, please feel free to contact: pengyuzhang@stu.xjtu.edu.cn
