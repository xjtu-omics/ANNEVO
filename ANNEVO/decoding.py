import argparse
from ANNEVO.src.post_process_decoding import predict_gff
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Decode gene structures based on predictions from a deep learning model.")
    parser.add_argument("--genome", required=True, help="Path to the genome file for decoding.")
    parser.add_argument("--model_prediction_path", required=True, help="Path to the model's predicted probability file.")
    parser.add_argument("--output", required=True, help="Path to the output GFF file.")
    parser.add_argument("--cpu_num", type=int, default=8, help="Number of CPU cores to use.")
    parser.add_argument("--average_threshold", type=float, default=0.1,
                        help="Minimum average probability threshold to consider a region as a potential gene.")
    parser.add_argument("--max_threshold", type=float, default=0.8,
                        help="Minimum CDS probability required at any site within a region to classify it as a potential gene.")
    parser.add_argument("--min_cds_length", type=int, default=60,
                        help="Minimum CDS length for a gene; genes shorter than this will be filtered out.")
    parser.add_argument("--min_cds_score", type=float, default=0.5,
                        help="Minimum average CDS score required for a gene; genes below this score will be filtered out.")
    parser.add_argument("--single_cds_score", type=float, default=0.7,
                        help="Threshold for high-confidence CDS regions; genes with all CDS scores below this will be filtered out.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    predict_gff(args.genome, args.model_prediction_path, args.output, args.cpu_num, args.average_threshold, args.max_threshold, args.min_cds_length, args.min_cds_score, args.single_cds_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene decoding took {elapsed_time} seconds")


if __name__ == "__main__":
    main()


