import argparse
from src.predict_nucleotide import nucleotide_prediction
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Predict per-nucleotide probabilities from a genome sequence.")
    parser.add_argument('--genome', required=True, help='Input genome FASTA file.')
    parser.add_argument('--model_path', required=True,
                        help='Path to the trained prediction model.')
    parser.add_argument('--genome_size_threshold', type=int, default=100 * 1024 * 1024,
                        help='Maximum cumulative contig size processed in one prediction batch. '
                             'When the cumulative size exceeds this threshold, prediction runs on the current chunk.')
    parser.add_argument('--model_prediction_path', type=str, default='model_prediction',
                        help='Output HDF5 path for model prediction results.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model inference.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for loading prediction data.')

    parser.add_argument('--window_size', type=int, default=30720,
                        help='Number of bases in each prediction window. This should match the value used during preprocessing and decoding.')
    parser.add_argument('--flank_length', type=int, default=5120,
                        help='Length of flanking sequence on each side of a prediction window. This should match the value used during preprocessing and decoding.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes predicted by the model.')
    args = parser.parse_args()

    if os.path.exists(args.model_prediction_path):
        raise FileExistsError(f"The file '{args.model_prediction_path}' already exists. Please delete it before running the prediction.")

    output_dir = os.path.dirname(args.model_prediction_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    nucleotide_prediction(args.genome, args.model_path, args.genome_size_threshold, args.num_workers, args.model_prediction_path,
                          args.batch_size, args.window_size, args.flank_length, args.num_classes)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The model prediction took {elapsed_time:.1f} seconds")


if __name__ == '__main__':
    main()
