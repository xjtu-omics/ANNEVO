## Performance comparison
Note: We have replaced blastp with diamond, which makes the evaluation much faster. We also no longer consider UTR regions and introns in them when evaluating the F1 score at the nucleotide level, because other tools such as BRAKER3 and ANNEVO v2 do not annotate UTR regions.  Therefore, we no longer perform error-checking masks on the regions in advance, but use the reference annotations as ground truth. This also affects the results by less than 0.5% in the performance evaluation of Mammalia, we show the original performance in brackets to demonstrate that the overall impact on the results is very small.  
### Performance on *Mammalia —— Sus_scrofa*
In our latest evaluation, we included a comparison with Tiberius (applicable to Mammalia only). However, as a strong caution, this performance comparison is not entirely fair. ANNEVO was trained using 21 species (with a total genome size of less than 50 Gb), whereas Tiberius was trained using 32 species (with a total genome size exceeding 80 Gb). Nevertheless, the optimized version of ANNEVO still achieved superior performance.  

| Model             | NT(CDS)-F1  | gene-F1 | BUSCO |
|:------------------|:-----------:|:-------:|------:|
| Augustus          | 68.6 (68.2) |  42.3   |  55.4 |
| BRAKER3           | 73.8 (73.5) |  66.7   |  75.9 |
| Helixer           | 84.6 (84.1) |  52.8   |  68.4 |
| Tiberius          |    91.6     |  78.8   |  92.5 |
| ANNEVO v1         | 89.3 (88.9) |  74.5   |  91.6 |
| ANNEVO v2         |    93.2     |  80.4   |  96.0 |

### Performance on *Vertebrate_other —— Danio_rerio* 

| Model     | NT(CDS)-F1 | gene-F1 | BUSCO |
|:----------|:----------:|:-------:|------:|
| Augustus  |    78.4    |  59.3   |  85.9 |
| BRAKER3   |    84.6    |  79.0   |  93.1 |
| Helixer   |    88.0    |  63.6   |  86.3 |
| ANNEVO v1 |    88.8    |  75.0   |  93.3 |
| ANNEVO v2 |    91.4    |  78.8   |  95.0 |

### Performance on *Invertebrate —— Drosophila_melanogaster* 

| Model     | NT(CDS)-F1 | gene-F1 | BUSCO |
|:----------|:----------:|:-------:|------:|
| Augustus  |    78.4    |  59.3   |  96.7 |
| BRAKER3   |    90.9    |  84.9   |  98.2 |
| Helixer   |    94.3    |  85.0   |  97.8 |
| ANNEVO v1 |    93.5    |  83.1   |  97.1 |
| ANNEVO v2 |    96.1    |  90.5   |  97.7 |

### Performance on *Fungi —— Aspergillus_oryzae* 
| Model     | NT(CDS)-F1 | gene-F1 | BUSCO |
|:----------|:----------:|:-------:|------:|
| Augustus  |    78.4    |  59.3   |  81.4 |
| BRAKER3   |    87.1    |  72.8   |  99.6 |
| Helixer   |    88.7    |  71.8   |  99.7 |
| ANNEVO v1 |    88.9    |  72.9   |  99.4 |
| ANNEVO v2 |    89.2    |  73.2   |  99.6 |

### Performance on *Plant —— Arabidopsis_thaliana and Zea_mays* 
#### Arabidopsis_thaliana
| Model     | NT(CDS)-F1 | gene-F1 | BUSCO |
|:----------|:----------:|:-------:|------:|
| Augustus  |    86.8    |  73.6   |  97.9 |
| BRAKER3   |    91.5    |  83.9   |  99.1 |
| Helixer   |    95.3    |  87.0   |  98.8 |
| ANNEVO v1 |    94.6    |  86.6   |  97.3 |
| ANNEVO v2 |    95.3    |  88.7   |  98.9 |
#### Zea_mays
| Model     | NT(CDS)-F1 | gene-F1 | BUSCO |
|:----------|:----------:|:-------:|------:|
| Augustus  |    51.4    |  31.7   |  55.9 |
| BRAKER3   |    81.8    |  70.1   |  97.4 |
| Helixer   |    71.3    |  61.0   |  94.3 |
| ANNEVO v1 |    82.7    |  72.1   |  94.5 |
| ANNEVO v2 |    87.5    |  78.7   |  96.2 |

## Computational time
Note that the computational time advantage will be more significant in plants, because ANNEVO only needs to decode potential gene regions, and the length of gene regions in plants is much smaller than that in mammals (gene length is shorter). BRAKER3 and others usually need to decode all regions.
### Sus scrofa (2.36G)
| Model                              | Prediction | Decoding | Total | 
|:-----------------------------------|:----------:|:--------:|------:|
| BRAKER3                            |    ---     |   ---    | 47.6h | 
| ANNEVO v1 (Step-by-step Execution) |   0.97h    |  1.97h   | 2.94h | 
| ANNEVO v2 (Step-by-step Execution) |   0.59h    |  0.66h   | 1.25h | 
| ANNEVO v2 (One-step Execution)     |    ---     |   ---    | 1.13h | 
### Arabidopsis thaliana (0.12G)
| Model                              | Prediction | Decoding |  Total | 
|:-----------------------------------|:----------:|:--------:|-------:|
| BRAKER3                            |    ---     |   ---    | 972min | 
| ANNEVO v1 (Step-by-step Execution) |   3.2min   |  2.6min  | 5.8min | 
| ANNEVO v2 (Step-by-step Execution) |   2.0min   |  1.5min  | 3.5min |
| ANNEVO v2 (One-step Execution)     |    ---     |   ---    | 3.4min | 
### Zea mays (2.06G)
| Model                              | Prediction | Decoding |  Total | 
|:-----------------------------------|:----------:|:--------:|-------:|
| BRAKER3                            |    ---     |   ---    | 102.7h | 
| ANNEVO v1 (Step-by-step Execution) |   0.71h    |  0.75h   |  1.46h | 
| ANNEVO v2 (Step-by-step Execution) |   0.49h    |  0.27h   |  0.76h |
| ANNEVO v2 (One-step Execution)     |    ---     |   ---    |  0.64h | 
