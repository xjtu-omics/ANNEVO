# ANNEVO update history

## 2026-08 — v2.3.3

- Open-sourced the training code.
- Added optional LZF compression for intermediate prediction H5 datasets. Compression is disabled by default and reduces prediction-file size without changing the final GFF output, but increases runtime due to the additional compression overhead.

## 2026-06 — v2.3.2

- Accelerated decoding with Numba.
- Further optimized resource management.
- Fixed the Actinopteri model upload. The previously uploaded file was an incorrect model version; other models were not affected.

## 2026-05 — v2.3.1

- Optimized memory usage, especially for fragmented genome assemblies.

## 2026-05 — v2.3.0

- Updated the model, training data, training strategy, and engineering implementation.
- Substantially improved annotation performance and execution speed.

## 2026-04 — v2.2.3

- Added a new plant model.
- Improved decoding speed by more than 30%.

## 2026-03 — v2.2.2

- Optimized the candidate-interval search logic used during decoding.

## 2026-01 — v2.2.1

- Released new Insecta and Mammalia models trained with the updated data-processing and training pipeline.

## 2025-10 — v2.2

- Optimized memory usage.

## 2025-07 — v2.1

- Introduced a new model architecture and training procedure.

## 2025-01 — v1.0

- Released ANNEVO for ab initio gene annotation.
