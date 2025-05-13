## Usage
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
