import argparse
from src.predict_nucleotide import nucleotide_prediction
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Predict nucleotide information.")
    parser.add_argument('--genome', required=True, help='The genome to be predicted.')
    parser.add_argument('--model_path', required=True,
                        help='Specify the path to the prediction model.')
    parser.add_argument('--genome_size_threshold', type=int, default=100 * 1024 * 1024,
                        help='Threshold for the total genome size per operation. '
                             'By default, whenever the cumulative size of contigs exceeds this threshold (e.g., 100 Mb), a prediction or decoding operation will be performed.')
    parser.add_argument('--model_prediction_path', type=str, default='model_prediction',
                        help='The storage path of the prediction results.')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in a batch.')
    parser.add_argument('--num_workers', type=int, default=16, help='The number of CPU cores to load data in parallel')

    parser.add_argument('--window_size', type=int, default=30720,
                        help='The number of bases in a window. Note: this parameter should be the same with it in data procession and gene decoding.')
    parser.add_argument('--flank_length', type=int, default=5120,
                        help='The length of flanking sequence. Note: this parameter should be the same with it in data procession and gene decoding.')
    parser.add_argument('--channels', type=int, default=64, help='The number of channels in Conv layer. Note: this parameter should be the same with it in gene decoding.')
    parser.add_argument('--dim_feedforward', type=int, default=768,
                        help='The dimension of linear layer in Transformer encoder. Note: this parameter should be the same with it in gene decoding.')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='The number of transformer encoder layer in each block. Note: this parameter should be the same with it in gene decoding.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='The number of attention heads in multi-heads attention. Note: this parameter should be the same with it in gene decoding.')
    parser.add_argument('--num_blocks', type=int, default=5, help='The number of Conv blocks. Note: this parameter should be the same with it in gene decoding.')
    parser.add_argument('--num_branches', type=int, default=8,
                        help='The number of simulated evolutionary branches. Note: this parameter should be the same with it in gene decoding.')
    args = parser.parse_args()

    if os.path.exists(args.model_prediction_path):
        raise FileExistsError(f"The file '{args.model_prediction_path}' already exists. Please delete it before running the prediction.")

    output_dir = os.path.dirname(args.model_prediction_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    nucleotide_prediction(args.genome, args.model_path, args.genome_size_threshold, args.num_workers, args.model_prediction_path, args.batch_size, args.window_size, args.flank_length, args.channels, args.dim_feedforward,
                          args.num_encoder_layers, args.num_heads, args.num_blocks, args.num_branches, num_classes=5)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The model prediction took {elapsed_time:.1f} seconds")


if __name__ == '__main__':
    main()
