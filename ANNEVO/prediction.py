import argparse
from ANNEVO.src.post_process_prediction import predict_proba_of_bases
import time
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Nucleotide prediction using ANNEVO. "
                                                 "Note: Do not modify parameters related to the model architecture, as this may cause errors. "
                                                 "These parameters are intended for customizing the model architecture when retraining for specific users. "
                                                 "For more details, refer to Retraining ANNEVO. "
                                                 "These parameters include: class_weights_base, class_weights_transition, class_weights_phases, window_size, flank_length"
                                                 "channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks, num_branches.")
    parser.add_argument('--genome', required=True, help='Path to the genome file for prediction.')
    parser.add_argument('--lineage', type=str, required=True,
                        choices=["Fungi", "Embryophyta", "Invertebrate", "Vertebrate_other", "Mammalia"],
                        help='Specify the lineage of the species to be predicted. This determines the deep learning model parameters used.'
                             'Options: Fungi, Embryophyta, Invertebrate, Vertebrate_other, Mammalia.')
    parser.add_argument('--chunk_num', type=int, default=5,
                        help='Maximum number of chunks to store the prediction results. Increasing this value can help prevent memory overflow.')
    parser.add_argument('--model_prediction_path', type=str, required=True,
                        help="Path to the model's prediction results.")
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples per batch.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of CPU cores used for parallel data loading.')
    parser.add_argument('--class_weights_base', type=float, nargs=4, default=[1, 1, 1, 1], help='Class weights for different base categories.')
    parser.add_argument('--class_weights_transition', type=float, nargs=2, default=[1, 1], help='Class weights for different transition categories.')
    parser.add_argument('--class_weights_phases', type=float, nargs=4, default=[1, 1, 1, 1], help='Class weights for different phase categories.')
    parser.add_argument('--window_size', type=int, default=20000, help='Number of bases in a window.')
    parser.add_argument('--flank_length', type=int, default=10000, help='Length of the flanking sequence.')
    parser.add_argument('--channels', type=int, default=16, help='Number of channels in the convolutional layer.')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Dimension of the linear layer in the Transformer encoder.')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of Transformer encoder layers per block.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in multi-head attention.')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of convolutional blocks.')
    parser.add_argument('--num_branches', type=int, default=8, help='Number of simulated evolutionary branches. This should match the value used in gene decoding.')
    args = parser.parse_args()

    num_classes_base = len(args.class_weights_base)
    num_classes_transition = len(args.class_weights_transition)
    num_classes_phases = len(args.class_weights_phases)
    if not os.path.exists(args.model_prediction_path):
        os.makedirs(args.model_prediction_path)
    if not os.path.exists(f'{args.model_prediction_path}/temp_genome_split'):
        os.makedirs(f'{args.model_prediction_path}/temp_genome_split')

    start_time = time.time()
    predict_proba_of_bases(args.genome, args.lineage, args.chunk_num, args.num_workers, args.model_prediction_path, args.batch_size, args.window_size, args.flank_length, args.channels, args.dim_feedforward,
                           args.num_encoder_layers, args.num_heads, args.num_blocks, args.num_branches, num_classes_base, num_classes_transition, num_classes_phases)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The model prediction took {elapsed_time} seconds")
    shutil.rmtree(f'{args.model_prediction_path}/temp_genome_split')


if __name__ == '__main__':
    main()
