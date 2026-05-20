## Notes
The evaluation was performed using gffcompare, excluding non-coding transcripts and UTR regions, and removing any transcripts with invalid start/stop codons or erroneous intron lengths, such as introns of length 1. 

For exon-level evaluation, we needed to ensure that each genomic position was assigned a unique label. Therefore, we used the longest transcript of each gene in the reference annotation as the ground truth. For locus-level evaluation, a predicted transcript from each tool was considered correct as long as it strictly matched any transcript of the same gene in the reference annotation, because ANNEVO, Helixer, and Tiberius currently do not directly support alternative splicing prediction.

For more details, please refer to the official gffcompare documentation: https://ccb.jhu.edu/software/stringtie/gffcompare.shtml. The command used was as follows:
```bash
gffcompare -r ${path_to_ref} ${path_to_pred} --no-exon-merge --strict-match
```

ANNEVO was intentionally trained without including model species; therefore, none of the test species appeared in either the training or validation sets. For Tiberius and Helixer, however, some of the test species may have been included in their training sets. From a user’s perspective, in some cases the practical performance of a method may be more relevant than a strictly controlled evaluation of the method itself. Therefore, we retained these results for the other methods and explicitly marked this point.

For Helixer, we used the best model recommended in their GitHub repository. For both Tiberius (v2.0.4) and ANNEVO (v2.3.0), we used the provided models corresponding to the relevant clades. 