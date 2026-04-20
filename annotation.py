import argparse
import time
import os
import tempfile
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run nucleotide prediction and gene structure decoding in one step.")
    parser.add_argument('--genome', required=True, help='Input genome FASTA file.')
    parser.add_argument('--model_path', required=True,
                        help='Path to the trained prediction model.')

    parser.add_argument("--output", required=True, help="Output GFF3 file path.")
    parser.add_argument("--threads", type=int, default=48, help="Number of CPU cores used for decoding.")
    parser.add_argument('--genome_size_threshold', type=int, default=100 * 1024 * 1024,
                        help='Maximum cumulative contig size processed in one prediction batch. '
                             'When the cumulative size exceeds this threshold, prediction runs on the current chunk.')
    parser.add_argument("--tmp_path", help="Directory for temporary intermediate files.")

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model inference.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for loading prediction data.')
    parser.add_argument('--window_size', type=int, default=30720,
                        help='Number of bases in each prediction window. This should match the value used during preprocessing and decoding.')
    parser.add_argument('--flank_length', type=int, default=5120,
                        help='Length of flanking sequence on each side of a prediction window. This should match the value used during preprocessing and decoding.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes predicted by the model.')
    parser.add_argument("--min_intron_length", type=int, default=1,
                        help="Minimum allowed length for CDS-associated introns during decoding.")
    parser.add_argument("--show_log", action="store_true", help="Show progress bars during decoding.")
    parser.add_argument("--boundary-aware", action="store_true",
                        help="Use boundary-aware decoding via decode_gene_structure2.")
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
    prediction_cmd = [
        sys.executable,
        prediction_script,
        "--genome", args.genome,
        "--model_path", args.model_path,
        "--genome_size_threshold", str(args.genome_size_threshold),
        "--model_prediction_path", model_prediction_path,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--window_size", str(args.window_size),
        "--flank_length", str(args.flank_length),
        "--num_classes", str(args.num_classes),
    ]
    subprocess.run(prediction_cmd, check=True)

    decoding_cmd = [
        sys.executable,
        decoding_script,
        "--genome", args.genome,
        "--model_prediction_path", model_prediction_path,
        "--output", args.output,
        "--threads", str(args.threads),
        "--min_intron_length", str(args.min_intron_length),
    ]
    if args.show_log:
        decoding_cmd.append("--show_log")
    if args.boundary_aware:
        decoding_cmd.append("--boundary-aware")
    subprocess.run(decoding_cmd, check=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene annotation took {elapsed_time:.1f} seconds")
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    main()
