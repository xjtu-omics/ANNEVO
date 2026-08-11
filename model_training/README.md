# Sequence model training

This directory contains the sequence-only training pipeline imported from ANNEVO v2.3.x and adapted to reuse the repository's existing `model/ANNEVO_seq.py` architecture.

Run commands from the repository root so Python can resolve both `model` and `model_training`.

## Reference annotation preprocessing

Before generating H5 files, use AGAT to retain the longest transcript for each gene:

```bash
agat_sp_keep_longest_isoform.pl \
  --gff reference.gff \
  --output reference.longest.gff
```

Use the resulting `reference.longest.gff` as the input to the subsequent annotation-cleaning and H5-generation steps.

> **Important:** Reference annotations may contain protein-coding transcripts with premature stop codons or introns that are shorter than the required minimum intron length. All such transcripts were filtered from the reference annotations used for ANNEVO training. Apply the same filtering to custom training references before generating H5 files; otherwise invalid gene structures will be introduced into the training labels.

Generate sequence/annotation H5 files:

```bash
bash model_training/scripts/generate_h5_data.sh \
  --genome genome.fa \
  --annotation annotation.gff \
  --output_prefix h5_data/train \
  --ig_ratio 0.2 \
  --length_config longer \
  --threads 8
```

Use `--ig_ratio -1` for validation data. The command creates `<prefix>_coding.h5` and `<prefix>_all.h5`.

## Length configuration

`--length_config` controls both the central prediction window and the flanking sequence supplied as model context:

| Configuration | Central window | Flank on each side | Total model input | Local pattern size |
| --- | ---: | ---: | ---: | ---: |
| `normal` | 30,720 bp | 5,120 bp | 40,960 bp | 32 bp |
| `longer` | 102,400 bp | 12,800 bp | 128,000 bp | 32 bp |

The annotation labels cover only the central window. The sequence stored in H5 includes both flanking regions, so its length is `central window + 2 * flank length`. Use the same `length_config` when generating H5 data and training the model.

## Combining multiple genomes

Multiple genomes can be written into the same pair of H5 files by running the command repeatedly with the same `--output_prefix`. Each run appends its chromosome samples to `<prefix>_coding.h5` and `<prefix>_all.h5`. If chromosome names repeat between genomes, a numeric suffix such as `__1` is added to keep every H5 group name unique.

For example:

```bash
output_prefix="h5_data/train"
genomes=("species_a.fa" "species_b.fa" "species_c.fa")
annotations=("species_a.gff" "species_b.gff" "species_c.gff")

# Start a new combined dataset. Omit this line when intentionally appending.
rm -f "${output_prefix}_coding.h5" "${output_prefix}_all.h5"

for index in "${!genomes[@]}"; do
  bash model_training/scripts/generate_h5_data.sh \
    --genome "${genomes[$index]}" \
    --annotation "${annotations[$index]}" \
    --output_prefix "${output_prefix}" \
    --ig_ratio 0.2 \
    --length_config normal \
    --threads 8
done
```

Remove the two existing output files before starting a new dataset; otherwise rerunning a genome appends duplicate samples.

Train on one GPU:

```bash
bash model_training/scripts/model_train_seq.sh \
  --train_h5_path h5_data/train_coding.h5 \
  --val_h5_path h5_data/val_coding.h5 \
  --train_h5_path_2 h5_data/train_all.h5 \
  --val_h5_path_2 h5_data/val_all.h5 \
  --length_config normal \
  --model_save_path saved_model/ANNEVO_seq.pt
```

For distributed training on multiple GPUs in one node, set `NPROC_PER_NODE` to the number of GPUs. For example, to train on four GPUs:

```bash
NPROC_PER_NODE=4 bash model_training/scripts/model_train_seq.sh \
  --train_h5_path h5_data/train_coding.h5 \
  --val_h5_path h5_data/val_coding.h5 \
  --train_h5_path_2 h5_data/train_all.h5 \
  --val_h5_path_2 h5_data/val_all.h5 \
  --length_config normal \
  --model_save_path saved_model/ANNEVO_seq.pt
```

To select specific GPUs, set `CUDA_VISIBLE_DEVICES` at the same time. The number of selected GPU IDs must match `NPROC_PER_NODE`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash model_training/scripts/model_train_seq.sh \
  --train_h5_path h5_data/train_coding.h5 \
  --val_h5_path h5_data/val_coding.h5 \
  --train_h5_path_2 h5_data/train_all.h5 \
  --val_h5_path_2 h5_data/val_all.h5 \
  --length_config normal \
  --model_save_path saved_model/ANNEVO_seq.pt
```

The script uses ordinary `python` when `NPROC_PER_NODE=1` and automatically switches to `torchrun` when the value is greater than one.

> **Training configuration reminder:** ANNEVO v2.3.x was trained with four GPUs. `BATCH_SIZE` in `model_train_seq.py` is the batch size per GPU, so the effective global batch size is `BATCH_SIZE * NPROC_PER_NODE`. Changing the number of GPUs without adjusting the per-GPU batch size changes both the global batch size and the number of optimizer/gradient-update steps in each epoch. This can make the learning-rate schedule and final training result differ from the v2.3.x reference training. For reproducibility, use four GPUs or adjust the batch size, learning rate, and warmup steps together.
