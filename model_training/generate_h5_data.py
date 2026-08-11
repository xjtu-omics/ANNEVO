import argparse
import random

import numpy as np

from model_training.datamodule.sequence_to_h5 import sequence_to_h5


LENGTH_CONFIG = {
    "longer": (102400, 12800),
    "normal": (30720, 5120),
}


def main():
    parser = argparse.ArgumentParser(description="Generate sequence-only H5 training datasets.")
    parser.add_argument("--genome", required=True, help="Genome FASTA path.")
    parser.add_argument("--annotation", required=True, help="Reference annotation GFF path.")
    parser.add_argument("--ig_ratio", type=float, required=True, help="Intergenic sampling ratio; -1 keeps all.")
    parser.add_argument("--output_prefix", required=True, help="Output prefix; writes *_coding.h5 and *_all.h5.")
    parser.add_argument("--length_config", required=True, choices=tuple(LENGTH_CONFIG))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    window_size, flank_length = LENGTH_CONFIG[args.length_config]
    outputs = sequence_to_h5(
        args.genome,
        args.annotation,
        args.output_prefix,
        args.threads,
        window_size,
        flank_length,
        args.ig_ratio,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
