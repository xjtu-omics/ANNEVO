import argparse
from src.gene_decoding import gene_structure_decoding
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Decode gene structure based on deep learning model's prediction.")
    parser.add_argument("-g", "--genome", required=True, help="Genome to be decoded.")
    parser.add_argument("-p", "--model_prediction_path", required=True,
                        help="Path to the probability predicted by the model.")
    parser.add_argument("-o", "--output", required=True, help="Output GFF file")
    parser.add_argument("-t", "--threads", type=int, default=48, help="Number of CPU cores used for decoding.")
    parser.add_argument("--show_log", action="store_true", help="Show decoding progress bars.")

    parser.add_argument("--min_intron_length", type=int, default=20,
                        help="Minimum intron length of CDS-associated intron groups.")
    parser.add_argument("--min_prot_length", type=int, default=100,
                        help="Predicted proteins shorter than this length are filtered with a higher confidence threshold.")
    args = parser.parse_args()
    AVE_THRESHOLD = 0.1
    MAX_THRESHOLD = 0.5
    MIN_CDS_LENGTH = args.min_prot_length * 3
    MIN_CDS_SCORE = 0.5
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    gene_structure_decoding(args.genome, args.model_prediction_path, args.output, args.threads,
                            AVE_THRESHOLD, MAX_THRESHOLD, MIN_CDS_LENGTH, MIN_CDS_SCORE, args.min_intron_length,
                            show_log=args.show_log)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene decoding took {elapsed_time:.1f} seconds")


if __name__ == "__main__":
    main()


