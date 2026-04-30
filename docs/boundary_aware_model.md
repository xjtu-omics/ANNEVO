# Usage
We added support for the new model through additional command-line arguments in the original command interface (only Magnoliopsida model).
## One-step Execution
Simply add `--num_classes 15 --boundary-aware`. We also recommend enabling the minimum intron length constraint with `--min_intron_length 20`.
```bash
python annotation.py --genome path_to_genome --model_path path_to_model --output path_to_gff --threads 48 --num_classes 15 --boundary-aware --min_intron_length 20
```
## Step-by-step Execution
Specifically, add `--num_classes 15` during the prediction stage and `--boundary-aware` during the decoding stage. As above, we also recommend enabling the minimum intron length constraint with `--min_intron_length 20`.
```bash
# Nucleotide prediction
python prediction.py --genome path_to_genome --model_path path_to_model --model_prediction_path path_to_save_predction --num_classes 15

# Gene structure decoding
python decoding.py --genome path_to_genome --model_prediction_path path_to_save_predction --output path_to_gff --threads 48 --boundary-aware --min_intron_length 20
```
# Evaluation and other notes
We have released a new plant model that uses a similar boundary-aware strategy to define label categories (Ref 1). However, the early stopping criterion, loss functions, model architecture, and decoding procedure remain the same as described in the ANNEVO (Ref 2). In particular, ANNEVO decoding is still driven entirely by the model probabilities, without any explicit modeling of length distributions. In addition, the data preprocessing pipeline was updated using a new strategy, which has not yet been published.  

**Because ANNEVO does not predict UTR annotations, we removed UTR regions from the reference annotation during evaluation, that is, non-coding exonic regions. Likewise, because ANNEVO focuses on protein-coding genes, regions associated with non-coding genes were not included in the evaluation.**

To enable a fair comparison with other methods, ANNEVO was deliberately trained with all model organisms excluded, such as Arabidopsis thaliana and rice. Evaluation was performed using gffcompare (Ref 3). For base-level and exon-level evaluation, the longest transcript of each reference annotation was used as the gold standard. For locus-level evaluation, the reference annotation containing all transcripts was used as the gold standard. We additionally disabled automatic exon merging and enabled strict matching. The command is shown below:
```
gffcompare -r ${path_to_ref} ${path_to_pred} --no-exon-merge --strict-match
```


| Species              | BUSCO (brassicales_odb12) | Base-recall | Base-precision | Exon-recall | Exon-precision | Locus-recall | Locus-precision |
|:---------------------|:-------------------------:|------------:|---------------:|------------:|---------------:|-------------:|----------------:|
| Arabidopsis_thaliana |           99.5            |        93.3 |           96.7 |        89.6 |           94.2 |         81.3 |            89.3 |
| **————————**         |   BUSCO (poales_odb12)    |    **————** |       **————** |    **————** |       **————** |     **————** |        **————** |  
| Oryza_sativa         |           99.5            |        91.4 |           91.6 |        90.5 |           89.6 |         80.1 |            78.5 |


For animal models such as Mammalia and Insecta, we have not released updated models in this version. Although we are aware that retraining the current models would further improve boundary accuracy, doing so would slow down development of subsequent releases. (We will release an updated model for fungi because its training cost is relatively small.)

In animals, introns are typically much longer, and therefore contribute far more than exons to the Viterbi decoding path. Therefore, accurate prediction of introns is more important. Given that since a longer context is needed to determine introns, we expect greater gains from directly releasing models with a longer effective inference length, rather than updating the current version again at this stage. For comparison, the effective core decoding region of ANNEVO is currently only about 30 kb, which is substantially shorter than the hundreds of kilobases used by some other methods. We therefore expect the next version of ANNEVO to achieve particularly notable improvements in animals, especially for long genes.

# Reference
[1] Gabriel L, Becker F, Hoff K J, et al. Tiberius: end-to-end deep learning with an HMM for gene prediction[J]. Bioinformatics, 2024, 40(12): btae685.  
[2] Zhang P, Xu T, Wang S, et al. Highly accurate ab initio gene annotation with ANNEVO[J]. Nature Methods, 2026: 1-9.  
[3] Pertea G, Pertea M. GFF utilities: GffRead and GffCompare[J]. F1000Research, 2020, 9: ISCB Comm J-304.