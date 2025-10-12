from data_process.genome2sample import generate_h5_file
import argparse
import time
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Process genomic data and save as h5 files.")
    parser.add_argument("--genome", required=True, help="Genome to be processed")
    parser.add_argument("--annotation", required=True, help="Genome annotation file used as label")
    parser.add_argument("--output_file", required=True, help="The prefix name of the output file (h5 format)")
    parser.add_argument("--threads", type=int, default=8, help="The number of CPU cores used simultaneously.")
    parser.add_argument("--window_size", type=int, default=30720, help="Core region.")
    parser.add_argument("--flank_length", type=int, default=5120, help="Flanking region")
    parser.add_argument("--keep_intergenic_sample", type=int, default=0, help="")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    generate_h5_file(args.genome, args.annotation, args.output_file, args.threads, args.window_size, args.flank_length, args.keep_intergenic_sample)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing genome took {elapsed_time} seconds")


if __name__ == "__main__":
    main()

