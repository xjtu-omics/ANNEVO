
# BUSCO databases
| Clade | Species | Database |
|---|---|---|
| Mammalia | *Homo sapiens* | mammalia_odb12 |
| Mammalia | *Rattus norvegicus* | mammalia_odb12 |
| Insecta | *Drosophila melanogaster* | drosophila_odb12 |
| Insecta | *Bombyx mori* | lepidoptera_odb12 |
| Fungi | *Saccharomyces cerevisiae* | saccharomycetaceae_odb12 |
| Fungi | *Candida albicans* | debaryomycetaceae_odb12 |
| Magnoliopsida | *Arabidopsis thaliana* | brassicales_odb12 |
| Magnoliopsida | *Oryza sativa* | poales_odb12 |
| Aves | *Gallus gallus* | aves_odb12 |
| Aves | *Melopsittacus undulatus* | aves_odb12 |
| Actinopteri | *Danio rerio* | actinopterygii_odb12 |
| Actinopteri | *Oryzias latipes* | actinopterygii_odb12 |

# Accuracy
### Helixer
| Clade | Species | Exon-Recall | Exon-Precision | Locus-Recall | Locus-Precision | BUSCO | Note           |
|---|---|---:|---:|---:|---:|---:|----------------|
| Mammalia | Homo_sapiens | 82.7 | 66.0 | 28.8 | 25.3 | 91.0 | *              |
| Mammalia | Rattus_norvegicus | 81.2 | 71.4 | 23.2 | 26.1 | 87.1 | *              |
| Insecta | Drosophila_melanogaster | 88.3 | 82.4 | 72.6 | 69.4 | 96.1 | * |
| Insecta | Bombyx_mori | 84.8 | 71.5 | 36.7 | 32.0 | 90.9 | In training set |
| Fungi | Saccharomyces_cerevisiae | 85.4 | 83.3 | 87.2 | 90.2 | 99.5 | In training set |
| Fungi | Candida_albicans | 92.4 | 87.8 | 92.3 | 91.2 | 97.5 | *              |
| Magnoliopsida | Arabidopsis_thaliana | 89.5 | 88.1 | 75.1 | 75.6 | 99.1 | In training set |
| Magnoliopsida | Oryza_sativa | 88.9 | 71.4 | 68.7 | 53.1 | 98.9 | In training set |
| Aves | Gallus_gallus | 84.7 | 70.2 | 30.3 | 28.4 | 92.0 | *              |
| Aves | Melopsittacus_undulatus | 86.7 | 69.1 | 30.7 | 26.8 | 92.0 | *              |
| Actinopteri | Danio_rerio | 84.2 | 69.7 | 27.6 | 22.0 | 80.6 | *              |
| Actinopteri | Oryzias_latipes | 84.8 | 73.2 | 28.6 | 24.3 | 85.3 | *              |
|  | Average | 86.1 | 75.3 | 50.2 | 47.0 | 92.5 |                |

Note: `*`indicates species mentioned by Helixer as part of its validation set. Helixer’s validation set does not consist of complete genomes, but rather 800  subsequences. Therefore, its performance on the validation set remains largely representative and is unlikely to bias the model toward the species included in the validation set.

### Tiberius
| Clade | Species | Exon-Recall | Exon-Precision | Locus-Recall | Locus-Precision | BUSCO | Note |
|---|---|---:|---:|---:|---:|---:|---|
| Mammalia | Homo_sapiens | 90.4 | 85.5 | 72.4 | 62.1 | 97.4 |  |
| Mammalia | Rattus_norvegicus | 89.2 | 88.8 | 67.5 | 63.0 | 96.3 |  |
| Insecta | Drosophila_melanogaster | 90.2 | 88.1 | 82.5 | 71.7 | 95.8 |  |
| Insecta | Bombyx_mori | 82.3 | 83.0 | 62.4 | 42.9 | 94.3 |  |
| Fungi | Saccharomyces_cerevisiae | 88.7 | 93.4 | 90.4 | 95.5 | 99.8 | In training set |
| Fungi | Candida_albicans | 94.0 | 93.8 | 93.9 | 94.7 | 97.7 | In training set |
| Magnoliopsida | Arabidopsis_thaliana | 89.6 | 94.0 | 80.5 | 88.2 | 99.0 |  |
| Magnoliopsida | Oryza_sativa | 87.4 | 88.2 | 74.6 | 71.3 | 95.9 |  |
| Aves | Gallus_gallus | 89.8 | 85.4 | 63.4 | 58.2 | 96.1 |  |
| Aves | Melopsittacus_undulatus | 92.3 | 84.4 | 64.0 | 54.6 | 95.9 |  |
| Actinopteri | Danio_rerio | 91.2 | 89.9 | 67.7 | 61.1 | 93.1 |  |
| Actinopteri | Oryzias_latipes | 91.9 | 90.4 | 68.6 | 61.9 | 93.7 |  |
|  | Average | 89.8 | 88.7 | 74.0 | 68.8 | 96.3 |  |

### ANNEVO
| Clade | Species | Exon-Recall | Exon-Precision | Locus-Recall | Locus-Precision | BUSCO | Note |
|---|---|---:|---:|---:|---:|---:|---|
| Mammalia | Homo_sapiens | 92.6 | 88.0 | 73.6 | 63.9 | 98.2 |  |
| Mammalia | Rattus_norvegicus | 91.9 | 89.4 | 70.1 | 65.0 | 97.3 |  |
| Insecta | Drosophila_melanogaster | 91.3 | 82.3 | 81.8 | 66.8 | 97.3 |  |
| Insecta | Bombyx_mori | 89.4 | 87.3 | 61.5 | 61.0 | 96.9 |  |
| Fungi | Saccharomyces_cerevisiae | 87.7 | 95.9 | 89.9 | 96.6 | 99.3 |  |
| Fungi | Candida_albicans | 92.9 | 93.9 | 93.2 | 94.3 | 97.6 |  |
| Magnoliopsida | Arabidopsis_thaliana | 89.5 | 94.4 | 80.8 | 90.0 | 99.4 |  |
| Magnoliopsida | Oryza_sativa | 90.4 | 90.0 | 79.9 | 79.5 | 99.5 |  |
| Aves | Gallus_gallus | 89.9 | 89.5 | 67.4 | 68.1 | 98.1 |  |
| Aves | Melopsittacus_undulatus | 93.4 | 87.3 | 68.5 | 61.5 | 97.7 |  |
| Actinopteri | Danio_rerio | 93.8 | 92.6 | 75.7 | 73.7 | 96.5 |  |
| Actinopteri | Oryzias_latipes | 94.2 | 92.3 | 73.5 | 71.0 | 96.2 |  |
|  | Average | 91.4 | 90.2 | 76.3 | 74.3 | 97.8 |  |
|


For all animal clades in ANNEVO, we used `--overlap_pred`, which slightly improved performance. We also provide the performance differences with and without this parameter for comparison: [Overlap_pred](overlap_pred.md)

# Annotation time and resource usage
We used a single RTX 4090 for the comparison, as this represents a hardware setup that is affordable for most GPU users. On the human genome, ANNEVO required 30 minutes for prediction and 22 minutes for decoding, for a total runtime of 52 minutes. In comparison, Tiberius required 141 minutes using a batch size of 8, which was the largest batch size that could run on an RTX 4090, whereas Helixer required 956 minutes.

Under the same batch size of 8, ANNEVO required 3.8 GB of GPU memory, while Tiberius required 22.5 GB and Helixer required 8.6 GB.

When no GPU was available, using a machine with 64 CPU cores, ANNEVO required 423 minutes for prediction and 22 minutes for decoding, for a total runtime of 445 minutes on the human genome. In comparison, Tiberius required 528 minutes and Helixer required 1381 minutes.