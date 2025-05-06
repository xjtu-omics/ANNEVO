from ANNEVO.data_process.data_preprocessing import create_dataset
import argparse
import time
import os


def is_fa_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error(f"The file {arg} does not exist!")
    elif not arg.endswith(('.fasta', '.fna', '.fa')):
        parser.error(f"The file {arg} is not a valid FASTA file! Allowed extensions: .fasta, .fna, .fa")
    else:
        return arg


def main():
    parser = argparse.ArgumentParser(description="Process the genome for deep learning models."
                                                 "When processing data for retraining, it is important to ensure that certain parameters remain consistent with subsequent training, "
                                                 "as these determine the input dimensions of the deep learning model."
                                                 "These parameters include: window_size and flank_length.")
    parser.add_argument("--genome", required=True, help="Path to the genome to be processed.", type=lambda x: is_fa_file(parser, x))
    parser.add_argument("--annotation", required=True, help="Annotation used as ground truth.")
    parser.add_argument("--output_file", required=True, help="Path to output file (h5 format)")
    parser.add_argument("--cpu_num", type=int, default=8, help="The number of CPUs in parallel.")
    parser.add_argument("--window_size", type=int, default=20000, help="Length of the core region.")
    parser.add_argument("--flank_length", type=int, default=10000, help="Length of each flanking region on the upstream and downstream.")
    parser.add_argument("--transition_weights", type=list, nargs=6, default=[16, 16, 16, 16, 16, 16],
                        help="Corresponding to the weights of intergenic-UTR, UTR-intergenic, UTR-CDS, CDS-UTR, CDS-intron (or UTR-intron), intron-CDS (or intron-UTR).")
    parser.add_argument("--mask_length", type=int, default=1000,
                        help="The default mask length when an error occurs.")
    parser.add_argument("--discard_all_intergenic", action='store_true',
                        help="Discard a window during training if all annotations in the reference database are intergenic regions. "
                             "Set this flag to discard all intergenic regions.")
    parser.add_argument("--no-discard_all_intergenic", dest='discard_all_intergenic', action='store_false', help="Keep all intergenic regions during training. "
                                                                                                                 "Set this flag to retain all intergenic regions.")
    parser.set_defaults(discard_all_intergenic=True)
    args = parser.parse_args()

    start_time = time.time()
    create_dataset(args.genome, args.annotation, args.output_file, args.cpu_num, args.window_size, args.flank_length,
                   args.transition_weights, args.mask_length, args.discard_all_intergenic)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing genome took {elapsed_time} seconds")


if __name__ == "__main__":
    main()

