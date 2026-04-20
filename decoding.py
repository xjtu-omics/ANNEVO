import argparse
from src.gene_decoding import gene_structure_decoding
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Decode gene structures from model-predicted nucleotide probabilities.")
    parser.add_argument("--genome", required=True, help="Input genome FASTA file.")
    parser.add_argument("--model_prediction_path", required=True, help="Input HDF5 file containing model prediction results.")
    parser.add_argument("--output", required=True, help="Output GFF3 file path.")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU cores used for decoding.")
    parser.add_argument("--show_log", action="store_true", help="Show progress bars during decoding.")
    parser.add_argument("--boundary-aware", action="store_true",
                        help="Use boundary-aware decoding via decode_gene_structure2.")

    parser.add_argument("--min_intron_length", type=int, default=1,
                        help="Minimum allowed length for CDS-associated introns during decoding.")
    args = parser.parse_args()
    AVE_THRESHOLD = 0.1
    MAX_THRESHOLD = 0.5
    MIN_CDS_LENGTH = 60
    MIN_CDS_SCORE = 0.5
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    gene_structure_decoding(args.genome, args.model_prediction_path, args.output, args.threads,
                            AVE_THRESHOLD, MAX_THRESHOLD, MIN_CDS_LENGTH, MIN_CDS_SCORE,
                            args.min_intron_length, show_log=args.show_log, boundary_aware=args.boundary_aware)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene decoding took {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()


