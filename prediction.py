import argparse
from src.predict_nucleotide import base_pred
import time
import os


def main():
    length_config = {
        'Mammalia': [102400, 12800, 32],
        'Insecta': [102400, 12800, 32],
        'Aves': [102400, 12800, 32],
        'Actinopteri': [102400, 12800, 32],
        'Magnoliopsida': [30720, 5120, 32],
        'Fungi': [30720, 5120, 32],
    }

    parser = argparse.ArgumentParser(description="Predict nucleotide information.")
    parser.add_argument('-g', '--genome', required=True, help='The genome to be predicted.')
    parser.add_argument('-m', '--model_path', required=True,
                        help='Specify the path to the prediction model.')
    parser.add_argument('-p', '--pred_path', required=True,
                        help='The storage path of the prediction results.')
    parser.add_argument('-l', "--lineage", required=True, choices=length_config.keys(),
                        help="Use lineage-specific config for seq_len")

    parser.add_argument('-s', '--genome_size_threshold', type=int, default=100,
                        help='Threshold for the total genome size per operation (M). '
                             'By default, whenever the cumulative size of contigs exceeds this threshold (e.g., 100 Mb), a prediction or decoding operation will be performed.')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in a batch.')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of CPU cores to load data in parallel')
    parser.add_argument('--overlap_pred', action='store_true',
                        help='Predict overlapping windows and average probabilities in overlapping output regions.')
    parser.add_argument('--comp', action='store_true',
                        help='Compress prediction datasets with LZF. Disabled by default.')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.pred_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    window_size, flank_length, local_pattern_size = length_config[args.lineage]
    total_len = window_size + 2 * flank_length
    if total_len % local_pattern_size != 0:
        raise ValueError(
            f"Lineage window config is invalid: window_size={window_size}, flank_length={flank_length}, "
            f"local_pattern_size={local_pattern_size}, total_len={total_len}. "
        )

    print(
        f"Prediction window/flank=({window_size}, {flank_length}), overlap_pred={args.overlap_pred}"
    )

    start_time = time.time()
    base_pred(
        args.genome,
        args.model_path,
        args.genome_size_threshold,
        args.pred_path,
        args.num_workers,
        args.batch_size,
        window_size,
        flank_length,
        local_pattern_size,
        args.lineage,
        overlap_pred=args.overlap_pred,
        comp=args.comp,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The model prediction took {elapsed_time:.1f} seconds")


if __name__ == '__main__':
    main()
