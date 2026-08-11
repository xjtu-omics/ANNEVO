import argparse
import time
import os
import tempfile
import shutil
import subprocess
import sys


def main():
    length_config = {
        'Mammalia': [102400, 12800, 32],
        'Insecta': [102400, 12800, 32],
        'Aves': [102400, 12800, 32],
        'Actinopteri': [102400, 12800, 32],
        'Magnoliopsida': [30720, 5120, 32],
        'Fungi': [30720, 5120, 32],
    }

    parser = argparse.ArgumentParser(description="Run nucleotide prediction and gene structure decoding in one step.")
    parser.add_argument('-g', '--genome', required=True, help='Input genome FASTA file.')
    parser.add_argument('-m', '--model_path', required=True,
                        help='Path to the trained prediction model.')
    parser.add_argument('-l', "--lineage", required=True, choices=length_config.keys(),
                        help="Use lineage-specific config for seq_len.")

    parser.add_argument('-o', "--output", required=True, help="Output GFF3 file path.")
    parser.add_argument("-t", "--threads", type=int, default=48, help="Number of CPU cores used for decoding.")
    parser.add_argument('-s', '--genome_size_threshold', type=int, default=100,
                        help='Threshold for the total genome size per operation (M).')
    parser.add_argument("--tmp_path", help="Directory for temporary intermediate files.")

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model inference.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for loading prediction data.')
    parser.add_argument('--overlap_pred', action='store_true',
                        help='Predict overlapping windows and average probabilities in overlapping output regions.')
    parser.add_argument('--comp', action='store_true',
                        help='Compress intermediate prediction datasets with LZF. Disabled by default.')
    parser.add_argument("--min_intron_length", type=int, default=20,
                        help="Minimum intron length of CDS-associated intron groups.")
    parser.add_argument("--min_prot_length", type=int, default=100,
                        help="Predicted proteins shorter than this length are filtered with a higher confidence threshold.")
    parser.add_argument("--show_log", action="store_true", help="Show decoding progress bars.")
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
        "-g", args.genome,
        "-m", args.model_path,
        "-l", args.lineage,
        "-s", str(args.genome_size_threshold),
        "-p", model_prediction_path,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ]
    if args.overlap_pred:
        prediction_cmd.append("--overlap_pred")
    if args.comp:
        prediction_cmd.append("--comp")
    subprocess.run(prediction_cmd, check=True)

    decoding_cmd = [
        sys.executable,
        decoding_script,
        "-g", args.genome,
        "-p", model_prediction_path,
        "-o", args.output,
        "-t", str(args.threads),
        "--min_intron_length", str(args.min_intron_length),
        "--min_prot_length", str(args.min_prot_length),
    ]
    if args.show_log:
        decoding_cmd.append("--show_log")
    subprocess.run(decoding_cmd, check=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The gene annotation took {elapsed_time:.1f} seconds")
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    main()
