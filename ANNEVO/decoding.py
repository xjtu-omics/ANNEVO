import argparse
from ANNEVO.src.post_process_decoding import predict_gff
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Decode gene structure based on deep learning model's prediction.")
    parser.add_argument("--genome", required=True, help="Path to the genome file for decoding.")
    parser.add_argument("--model_prediction_path", required=True, help="Path to the model's predicted probability file.")
    parser.add_argument("--output", required=True, help="Path to the output GFF file.")
    parser.add_argument('--lineage', type=str, required=True,
                        choices=["Fungi", "Embryophyta", "Invertebrate", "Vertebrate_other", "Mammalia"],
                        help='Specify the lineage of the species to be decoded.'
                             'Options: Fungi, Embryophyta, Invertebrate, Vertebrate_other, Mammalia.')
    parser.add_argument("--threads", type=int, default=8, help="Number of threads used simultaneously.")
    parser.add_argument("--average_threshold", type=float, default=0.1,
                        help="Minimum average probability threshold to consider a region as a potential gene.")
    parser.add_argument("--max_threshold", type=float, default=0.5,
                        help="Minimum CDS probability required at any site within a region to classify it as a potential gene.")
    parser.add_argument("--min_cds_length", type=int, default=100,
                        help="Minimum CDS length for a gene; genes shorter than this will be filtered out.")
    parser.add_argument("--min_cds_score", type=float,
                        help="The lowest CDS score in a gene. Genes with CDS score below this value will be filtered out. "
                             "This will serve as a parameter to balance completeness and false positives. "
                             "If specified, this score will be used as a filter for gene confidence scores.")
    parser.add_argument("--min_intron_length", type=int, default=2,
                        help="Minimum intron length of CDS-associated intron groups")
    parser.add_argument("--allow_start_end_splice", type=int, default=0,
                        help="Whether to allow start and stop codons to be interrupted by splicing. Setting this parameter to 1 will activate this option.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_time = time.time()
    predict_gff(args.genome, args.model_prediction_path, args.output, args.threads, args.average_threshold, args.max_threshold, args.min_cds_length, args.min_cds_score,
                args.min_intron_length, args.lineage, args.allow_start_end_splice)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene decoding took {elapsed_time} seconds")


if __name__ == "__main__":
    main()


