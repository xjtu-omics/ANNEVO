import argparse
import time
import os
import tempfile
import shutil


def main():
    parser = argparse.ArgumentParser(description="One-step annotation")
    parser.add_argument('--genome', required=True, help='The genome to be predicted.')
    parser.add_argument('--model_path', required=True,
                        help='Specify the path to the prediction model.')

    parser.add_argument("--output", required=True, help="Output GFF file")
    parser.add_argument("--threads", type=int, default=48, help="Number of CPU cores used simultaneously.")
    parser.add_argument('--genome_size_threshold', type=int, default=100 * 1024 * 1024,
                        help='Threshold for the total genome size per operation. '
                             'By default, whenever the cumulative size of contigs exceeds this threshold (e.g., 100 Mb), a prediction or decoding operation will be performed.')
    parser.add_argument("--tmp_path", help="Path to save temporary intermediate files")

    parser.add_argument('--batch_size', type=int, default=32, help='The number of samples in a batch.')
    parser.add_argument('--num_workers', type=int, default=8, help='The number of CPU cores to load data in parallel')
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
    parser.add_argument("--at_ac_splicing", type=int, default=0,
                        help="Enable AT-AC splicing mode")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.tmp_path:
        tmp_folder = tempfile.mkdtemp(prefix="tmp_", dir=f"{args.tmp_path}")
    else:
        os.makedirs("tmp", exist_ok=True)
        tmp_folder = tempfile.mkdtemp(prefix="tmp_", dir="./tmp")
    model_prediction_path = os.path.join(tmp_folder, "model_prediction.h5")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    prediction_script = os.path.join(BASE_DIR, "prediction.py")
    decoding_script = os.path.join(BASE_DIR, "decoding.py")
    start_time = time.time()
    cmd = (
        f"python {prediction_script} "
        f"--genome {args.genome} "
        f"--model_path {args.model_path} "
        f"--genome_size_threshold {args.genome_size_threshold} "
        f"--model_prediction_path {model_prediction_path} "
        f"--batch_size {args.batch_size} "
        f"--num_workers {args.num_workers} "
        f"--window_size {args.window_size} "
        f"--flank_length {args.flank_length} "
        f"--channels {args.channels} "
        f"--dim_feedforward {args.dim_feedforward} "
        f"--num_encoder_layers {args.num_encoder_layers} "
        f"--num_heads {args.num_heads} "
        f"--num_blocks {args.num_blocks} "
        f"--num_branches {args.num_branches}"
    )
    os.system(cmd)

    cmd = (
        f"python {decoding_script} "
        f"--genome {args.genome} "
        f"--model_prediction_path {model_prediction_path} "
        f"--genome_size_threshold {args.genome_size_threshold} "
        f"--output {args.output} "
        f"--threads {args.threads} "
        f"--average_threshold {args.average_threshold} "
        f"--max_threshold {args.max_threshold} "
        f"--min_cds_length {args.min_cds_length} "
        f"--min_cds_score {args.min_cds_score} "
        f"--min_intron_length {args.min_intron_length} "
        f"--at_ac_splicing {args.at_ac_splicing}"
    )
    os.system(cmd)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene annotation took {elapsed_time:.1f} seconds")
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    main()
