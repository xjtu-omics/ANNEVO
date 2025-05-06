from src import utils
import argparse
from src.gene_decoding import gene_structure_decoding
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Decode gene structure based on deep learning model's prediction.")
    parser.add_argument("--genome", required=True, help="Genome to be decoded.")
    parser.add_argument("--model_prediction_path", required=True, help="Path to the probability predicted by the model.")
    parser.add_argument("--output", required=True, help="Output GFF file")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU cores used simultaneously.")
    parser.add_argument("--average_threshold", type=float, default=0.1,
                        help="The minimum threshold of average probability when judging whether a region is a potential gene region.")
    parser.add_argument("--max_threshold", type=float, default=0.5,
                        help="When judging whether a region is a potential gene region, the CDS probability of at least one site in the region needs to meet this threshold.")
    parser.add_argument("--min_cds_length", type=int, default=60,
                        help="The shortest CDS length in a gene. Genes with CDS lengths below this value will be filtered out.")
    parser.add_argument("--min_cds_score", type=float, default=0.5,
                        help="The lowest CDS score in a gene. Genes with CDS score below this value will be filtered out. "
                             "This will serve as a parameter to balance completeness and false positives. "
                             "If specified, this score will be used as a filter for gene confidence scores.")
    parser.add_argument("--min_intron_length", type=int, default=1,
                        help="Minimum intron length of CDS-associated intron groups")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    gene_structure_decoding(args.genome, args.model_prediction_path, args.output, args.threads, args.average_threshold, args.max_threshold, args.min_cds_length, args.min_cds_score,
                            args.min_intron_length)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene decoding took {elapsed_time} seconds")


if __name__ == "__main__":
    main()


