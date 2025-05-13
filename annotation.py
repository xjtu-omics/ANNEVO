import argparse
from src.gene_annotation import pred_and_decode
import time
import os


def main():
    parser = argparse.ArgumentParser(description="Train deep learning model")
    parser.add_argument('--genome', required=True, help='The genome to be predicted.')
    parser.add_argument('--lineage', type=str, required=True,
                        help='The lineage of deep learning model. Optional: vertebrate_mammalian, plant')

    parser.add_argument("--output", required=True, help="Output GFF file")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU cores used simultaneously.")
    parser.add_argument('--chunk_num', type=int, default=5,
                        help='The maximum number of blocks to store the predicted file. Split the prediction probability file into chunks to avoid memory overflow. '
                             'When memory is insufficient, this value can be increased.')

    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in a batch.')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of CPU cores to load data in parallel')
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
    pred_and_decode(args.genome, args.lineage, args.chunk_num, args.output, args.threads, args.num_workers, args.batch_size, args.window_size, args.flank_length, args.channels, args.dim_feedforward,
                    args.num_encoder_layers, args.num_heads, args.num_blocks, args.num_branches, args.average_threshold, args.max_threshold, args.min_cds_length, args.min_cds_score,
                    args.min_intron_length, num_classes=5)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The model prediction took {elapsed_time} seconds")


if __name__ == '__main__':
    main()
