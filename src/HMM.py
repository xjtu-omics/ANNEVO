import numpy as np
from tqdm import tqdm


def define_state(intron_group, min_intron_length, min_intron_length_rare):
    states = ['intergenic', 'start0', 'start1', 'start2', 'CDS0', 'CDS0_T', 'CDS1', 'CDS1_TA', 'CDS1_TG', 'CDS2', 'end0', 'end1_TA', 'end1_TG', 'end2',
              'CDS0_ext', 'CDS0_T_ext', 'CDS1_ext', 'CDS1_TA_ext', 'CDS1_TG_ext', 'CDS2_ext']

    intron_columns = []
    splice_mode = ['GT_AG']
    for state in intron_group:
        for splice in splice_mode:
            states.append(f'{state}_{splice}_DSS_0')
            states.append(f'{state}_{splice}_DSS_1')
            states.append(f'{state}_{splice}_ASS_0')
            states.append(f'{state}_{splice}_ASS_1')
            intron_columns.append(f'{state}_{splice}_DSS_0')
            intron_columns.append(f'{state}_{splice}_DSS_1')
            intron_columns.append(f'{state}_{splice}_ASS_0')
            intron_columns.append(f'{state}_{splice}_ASS_1')
            for i in range(min_intron_length):
                states.append(f'{state}_GT_AG_intron_{i}')
                intron_columns.append(f'{state}_GT_AG_intron_{i}')

    states_to_num = {s: i for i, s in enumerate(states)}
    num_states = len(states_to_num)
    phase_0_columns = ['start0', 'end0', 'CDS0', 'CDS0_T', 'CDS0_ext', 'CDS0_T_ext']
    phase_2_columns = ['start1', 'end1_TG', 'end1_TA', 'CDS1', 'CDS1_TA', 'CDS1_TG', 'CDS1_ext', 'CDS1_TA_ext', 'CDS1_TG_ext']
    phase_1_columns = ['start2', 'end2', 'CDS2', 'CDS2_ext']

    phase_0_columns = [states_to_num[state] for state in phase_0_columns]
    phase_1_columns = [states_to_num[state] for state in phase_1_columns]
    phase_2_columns = [states_to_num[state] for state in phase_2_columns]
    intron_columns = [states_to_num[state] for state in intron_columns]
    return states_to_num, num_states, phase_0_columns, phase_1_columns, phase_2_columns, intron_columns


def set_transition_matrix_four_bases(init_transition_matrix, states_to_num, intron_group, min_intron_length, min_intron_length_rare,
                                     exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty, intron_quit_penalty):
    transition_matrix_A = init_transition_matrix.copy()
    transition_matrix_G = init_transition_matrix.copy()
    transition_matrix_C = init_transition_matrix.copy()
    transition_matrix_T = init_transition_matrix.copy()
    transition_matrix_other = init_transition_matrix.copy()

    # current base == A
    transition_matrix_A[states_to_num['CDS0_T'], states_to_num['CDS1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['CDS0_T_ext'], states_to_num['CDS1_TA_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_A[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num['CDS1_TA_ext']] = 0
    for state in intron_group:
        transition_matrix_A[states_to_num[f'{state}_GT_AG_intron_{min_intron_length - 1}'], states_to_num[f'{state}_GT_AG_ASS_0']] = intron_quit_penalty
    transition_matrix_A[states_to_num[f'end0'], states_to_num[f'end1_TA']] = 0
    transition_matrix_A[states_to_num[f'end1_TA'], states_to_num[f'end2']] = 0
    transition_matrix_A[states_to_num[f'end1_TG'], states_to_num[f'end2']] = 0
    transition_matrix_A[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['CDS2_ext'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_A[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num[f'CDS0_ext']] = 0

    # current base == T
    for splice in ['GT_AG']:
        transition_matrix_T[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num['CDS0_T_ext']] = 0
    for state in intron_group:
        transition_matrix_T[states_to_num[f'{state}_GT_AG_DSS_0'], states_to_num[f'{state}_GT_AG_DSS_1']] = 0

    transition_matrix_T[states_to_num[f'CDS2_ext'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num[f'CDS2'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num[f'start2'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num[f'CDS2'], states_to_num[f'end0']] = 0
    transition_matrix_T[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS1_TG_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_T[states_to_num[f'CDS1_TG_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0
    transition_matrix_T[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS0_T_ext'], states_to_num['CDS1_ext']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS1_TA_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_T[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num[f'CDS1_ext']] = 0
        transition_matrix_T[states_to_num[F'CDS1_TA_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0

    # current base == G
    transition_matrix_G[states_to_num['CDS0_T'], states_to_num['CDS1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS0_T_ext'], states_to_num['CDS1_TG_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_G[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num['CDS1_TG_ext']] = 0
    for state in intron_group:
        transition_matrix_G[states_to_num[f'{state}_GT_AG_ASS_0'], states_to_num[f'{state}_GT_AG_ASS_1']] = 0
        transition_matrix_G[states_to_num[f'{state}'], states_to_num[f'{state}_GT_AG_DSS_0']] = exon_quit_penalty
    transition_matrix_G[states_to_num[f'end1_TA'], states_to_num[f'end2']] = 0
    transition_matrix_G[states_to_num[f'end0'], states_to_num[f'end1_TG']] = 0
    transition_matrix_G[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS1_TG_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_G[states_to_num[f'CDS1_TG_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0
    transition_matrix_G[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS2_ext'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_G[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num[f'CDS0_ext']] = 0

    # current base == C
    transition_matrix_C[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS1_TG_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_C[states_to_num[f'CDS1_TG_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0
    transition_matrix_C[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS2_ext'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_C[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num[f'CDS0_ext']] = 0
    transition_matrix_C[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS0_T_ext'], states_to_num['CDS1_ext']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS1_TA_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_C[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num[f'CDS1_ext']] = 0
        transition_matrix_C[states_to_num[F'CDS1_TA_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0

    transition_matrix_other[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_other[states_to_num['CDS1_TG_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_other[states_to_num[f'CDS1_TG_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0
    transition_matrix_other[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_other[states_to_num['CDS2_ext'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_other[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_other[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num[f'CDS0_ext']] = 0
    transition_matrix_other[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_other[states_to_num['CDS0_T_ext'], states_to_num['CDS1_ext']] = exon_sustain_penalty
    transition_matrix_other[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_other[states_to_num['CDS1_TA_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    for splice in ['GT_AG']:
        transition_matrix_other[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num[f'CDS1_ext']] = 0
        transition_matrix_other[states_to_num[F'CDS1_TA_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0
    return transition_matrix_A, transition_matrix_G, transition_matrix_C, transition_matrix_T, transition_matrix_other


def set_transition_matrix(transition_matrix, states_to_num, intron_group, min_intron_length, min_intron_length_rare, current_base,
                          exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty, intron_quit_penalty, threshold, current_emit_penalty):
    if current_base == 'A':
        if current_emit_penalty[states_to_num['start0']] < threshold:
            transition_matrix[states_to_num[f'intergenic'], states_to_num[f'start0']] = current_emit_penalty[states_to_num['start0']] * 10
        else:
            transition_matrix[states_to_num[f'intergenic'], states_to_num[f'start0']] = 0

    elif current_base == 'T':
        if current_emit_penalty[states_to_num['start1']] < threshold:
            transition_matrix[states_to_num[f'start0'], states_to_num[f'start1']] = current_emit_penalty[states_to_num['start1']] * 10
        else:
            transition_matrix[states_to_num[f'start0'], states_to_num[f'start1']] = 0
    elif current_base == 'G':
        if current_emit_penalty[states_to_num['start2']] < threshold:
            transition_matrix[states_to_num[f'start1'], states_to_num[f'start2']] = current_emit_penalty[states_to_num['start2']] * 10
        else:
            transition_matrix[states_to_num[f'start1'], states_to_num[f'start2']] = 0
    elif current_base == 'C':
        for state in intron_group:
            transition_matrix[states_to_num[f'{state}_GT_AG_DSS_0'], states_to_num[f'{state}_GT_AG_DSS_1']] = current_emit_penalty[states_to_num[f'{state}_GT_AG_DSS_1']] * 1000

    return transition_matrix

# def set_transition_matrix(transition_matrix, states_to_num, intron_group, min_intron_length, min_intron_length_rare, current_base,
#                           exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty, intron_quit_penalty, threshold, current_emit_penalty):
#     if current_base == 'A':
#         transition_matrix[states_to_num['CDS0_T'], states_to_num['CDS1_TA']] = exon_sustain_penalty
#         transition_matrix[states_to_num['CDS0_T_ext'], states_to_num['CDS1_TA_ext']] = exon_sustain_penalty
#         for splice in ['GT_AG']:
#             transition_matrix[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num['CDS1_TA_ext']] = 0
#         for state in intron_group:
#             transition_matrix[states_to_num[f'{state}_GT_AG_intron_{min_intron_length - 1}'], states_to_num[f'{state}_GT_AG_ASS_0']] = intron_quit_penalty
#             # if current_emit_penalty[states_to_num[f'{state}_GT_AG_ASS_0']] > threshold:
#             #     transition_matrix[states_to_num[f'{state}_GT_AG_intron_{min_intron_length - 1}'], states_to_num[f'{state}_GT_AG_ASS_0']] = intron_quit_penalty
#         # if current_emit_penalty[states_to_num['start0']] > threshold:
#         #     transition_matrix[states_to_num[f'intergenic'], states_to_num[f'start0']] = 0
#         if current_emit_penalty[states_to_num['start0']] < threshold:
#             transition_matrix[states_to_num[f'intergenic'], states_to_num[f'start0']] = current_emit_penalty[states_to_num['start0']] * 10
#         else:
#             transition_matrix[states_to_num[f'intergenic'], states_to_num[f'start0']] = 0
#         transition_matrix[states_to_num[f'end0'], states_to_num[f'end1_TA']] = 0
#         transition_matrix[states_to_num[f'end1_TA'], states_to_num[f'end2']] = 0
#         transition_matrix[states_to_num[f'end1_TG'], states_to_num[f'end2']] = 0
#
#     elif current_base == 'T':
#         for splice in ['GT_AG']:
#             transition_matrix[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num['CDS0_T_ext']] = 0
#             # transition_matrix[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num['end0']] = 0
#         for state in intron_group:
#             transition_matrix[states_to_num[f'{state}_GT_AG_DSS_0'], states_to_num[f'{state}_GT_AG_DSS_1']] = 0
#             # if current_emit_penalty[states_to_num[f'{state}_GT_AG_DSS_1']] > threshold:
#             #     transition_matrix[states_to_num[f'{state}_GT_AG_DSS_0'], states_to_num[f'{state}_GT_AG_DSS_1']] = 0
#
#         transition_matrix[states_to_num[f'CDS2_ext'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
#         transition_matrix[states_to_num[f'CDS2'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
#         transition_matrix[states_to_num[f'start2'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
#         # if current_emit_penalty[states_to_num['start1']] > threshold:
#         #     transition_matrix[states_to_num[f'start0'], states_to_num[f'start1']] = 0
#         if current_emit_penalty[states_to_num['start1']] < threshold:
#             transition_matrix[states_to_num[f'start0'], states_to_num[f'start1']] = current_emit_penalty[states_to_num['start1']] * 10
#         else:
#             transition_matrix[states_to_num[f'start0'], states_to_num[f'start1']] = 0
#         transition_matrix[states_to_num[f'CDS2'], states_to_num[f'end0']] = 0
#     elif current_base == 'G':
#         transition_matrix[states_to_num['CDS0_T'], states_to_num['CDS1_TG']] = exon_sustain_penalty
#         transition_matrix[states_to_num['CDS0_T_ext'], states_to_num['CDS1_TG_ext']] = exon_sustain_penalty
#         for splice in ['GT_AG']:
#             transition_matrix[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num['CDS1_TG_ext']] = 0
#         for state in intron_group:
#             transition_matrix[states_to_num[f'{state}_GT_AG_ASS_0'], states_to_num[f'{state}_GT_AG_ASS_1']] = 0
#             transition_matrix[states_to_num[f'{state}'], states_to_num[f'{state}_GT_AG_DSS_0']] = exon_quit_penalty
#             # if current_emit_penalty[states_to_num[f'{state}_GT_AG_ASS_1']] > threshold:
#             #     transition_matrix[states_to_num[f'{state}_GT_AG_ASS_0'], states_to_num[f'{state}_GT_AG_ASS_1']] = 0
#             # if current_emit_penalty[states_to_num[f'{state}_GT_AG_DSS_0']] > threshold:
#             #     transition_matrix[states_to_num[f'{state}'], states_to_num[f'{state}_GT_AG_DSS_0']] = exon_quit_penalty
#         # if current_emit_penalty[states_to_num['start2']] > threshold:
#         #     transition_matrix[states_to_num[f'start1'], states_to_num[f'start2']] = 0
#         if current_emit_penalty[states_to_num['start2']] < threshold:
#             transition_matrix[states_to_num[f'start1'], states_to_num[f'start2']] = current_emit_penalty[states_to_num['start2']] * 10
#         else:
#             transition_matrix[states_to_num[f'start1'], states_to_num[f'start2']] = 0
#         # transition_matrix[states_to_num[f'start2'], states_to_num[f'CDS2_GT_AG_DSS_0']] = 0
#         transition_matrix[states_to_num[f'end1_TA'], states_to_num[f'end2']] = 0
#         transition_matrix[states_to_num[f'end0'], states_to_num[f'end1_TG']] = 0
#     elif current_base == 'C':
#         for state in intron_group:
#             transition_matrix[states_to_num[f'{state}_GT_AG_DSS_0'], states_to_num[f'{state}_GT_AG_DSS_1']] = current_emit_penalty[states_to_num[f'{state}_GT_AG_DSS_1']] * 1000
#
#     if current_base != 'A':
#         transition_matrix[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
#         transition_matrix[states_to_num['CDS1_TG_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
#         for splice in ['GT_AG']:
#             transition_matrix[states_to_num[f'CDS1_TG_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0
#     if current_base != 'T':
#         transition_matrix[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
#         transition_matrix[states_to_num['CDS2_ext'], states_to_num['CDS0']] = exon_sustain_penalty
#         transition_matrix[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
#         for splice in ['GT_AG']:
#             transition_matrix[states_to_num[f'CDS2_{splice}_ASS_1'], states_to_num[f'CDS0_ext']] = 0
#     if current_base not in ['A', 'G']:
#         transition_matrix[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
#         transition_matrix[states_to_num['CDS0_T_ext'], states_to_num['CDS1_ext']] = exon_sustain_penalty
#         transition_matrix[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
#         transition_matrix[states_to_num['CDS1_TA_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
#         for splice in ['GT_AG']:
#             transition_matrix[states_to_num[f'CDS0_T_{splice}_ASS_1'], states_to_num[f'CDS1_ext']] = 0
#             transition_matrix[states_to_num[F'CDS1_TA_{splice}_ASS_1'], states_to_num['CDS2_ext']] = 0
#
#     return transition_matrix


def set_transition_matrix_common_state(transition_matrix, states_to_num, intron_group, min_intron_length, min_intron_length_rare,
                                       exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty, intron_quit_penalty):
    transition_matrix[states_to_num['intergenic'], states_to_num['intergenic']] = 0
    transition_matrix[states_to_num['CDS0'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix[states_to_num['CDS1'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix[states_to_num['CDS0_ext'], states_to_num['CDS1_ext']] = exon_sustain_penalty
    transition_matrix[states_to_num['CDS1_ext'], states_to_num['CDS2_ext']] = exon_sustain_penalty
    transition_matrix[states_to_num['end2'], states_to_num['intergenic']] = 0

    for splice in ['GT_AG']:
        transition_matrix[states_to_num[f'CDS0_{splice}_ASS_1'], states_to_num[f'CDS1_ext']] = 0
        transition_matrix[states_to_num[f'CDS1_{splice}_ASS_1'], states_to_num[f'CDS2_ext']] = 0
    for state in intron_group:
        transition_matrix[states_to_num[f'{state}_GT_AG_DSS_1'], states_to_num[f'{state}_GT_AG_intron_0']] = 0
        transition_matrix[states_to_num[f'{state}_GT_AG_intron_{min_intron_length - 1}'], states_to_num[f'{state}_GT_AG_intron_{min_intron_length - 1}']] = intron_sustain_penalty
        if min_intron_length > 1:
            for i in range(min_intron_length - 1):
                transition_matrix[states_to_num[f'{state}_GT_AG_intron_{i}'], states_to_num[f'{state}_GT_AG_intron_{i + 1}']] = 0
    return transition_matrix


def viterbi_decoding(predictions, sequence, states_to_num, num_states, phase_0_columns, phase_1_columns, phase_2_columns, intron_columns, intron_group, min_intron_length_rare,
                     min_intron_length, expect_exon=None, expect_intron=None):
    """
    Decoding gene structure using viterbi algorithm.
    """
    np.seterr(divide='ignore', invalid='ignore')
    epsilon = 1e-64
    extra_penalty_coefficient = 5
    predictions[predictions < epsilon] = epsilon

    seq_length = predictions.shape[0]
    log_emit_probs = np.zeros((seq_length, num_states))
    threshold = np.log(0.1)
    # threshold = -np.inf
    states = list(range(num_states))

    # max_vals = np.max(predictions[:, 1:4], axis=1, keepdims=True)  # shape=[N, 1]
    # thresholds = max_vals / 2
    # mask = predictions[:, 1:4] < thresholds
    # predictions[:, 1:4] = np.where(mask, thresholds, predictions[:, 1:4])

    log_emit_probs[:, states_to_num['intergenic']] = np.log(predictions[:, 0])
    log_emit_probs[:, phase_0_columns] = np.log(predictions[:, 1][:, np.newaxis])
    log_emit_probs[:, phase_2_columns] = np.log(predictions[:, 3][:, np.newaxis])
    log_emit_probs[:, phase_1_columns] = np.log(predictions[:, 2][:, np.newaxis])
    log_emit_probs[:, intron_columns] = np.log(predictions[:, 4][:, np.newaxis])

    # cds_columns = phase_0_columns + phase_2_columns + phase_1_columns
    # log_emit_probs[:, cds_columns] = np.where(
    #     log_emit_probs[:, cds_columns] < 0.01,
    #     log_emit_probs[:, cds_columns] * extra_penalty_coefficient,
    #     log_emit_probs[:, cds_columns]
    # )

    if expect_exon:
        exon_sustain_penalty = np.log(1 - 1 / expect_exon)
        exon_quit_penalty = np.log(1 / expect_exon)
    else:
        exon_sustain_penalty = 0
        exon_quit_penalty = 0
    if expect_intron:
        intron_sustain_penalty = np.log(1 - 1 / expect_intron)
        intron_quit_penalty = np.log(1 / expect_intron)
    else:
        intron_sustain_penalty = 0
        intron_quit_penalty = 0

    init_transition_matrix = np.full((num_states, num_states), -np.inf)
    init_transition_matrix = set_transition_matrix_common_state(init_transition_matrix, states_to_num, intron_group, min_intron_length, min_intron_length_rare,
                                                                exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty, intron_quit_penalty)
    (transition_matrix_A, transition_matrix_G, transition_matrix_C,
     transition_matrix_T, transition_matrix_other) = set_transition_matrix_four_bases(init_transition_matrix, states_to_num, intron_group, min_intron_length, min_intron_length_rare,
                                                                                      exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty, intron_quit_penalty)
    path = np.zeros((seq_length, num_states), dtype=int)
    dp = np.full((seq_length, num_states), -np.inf)
    dp[0, 0] = 0
    # dp[0, :] = log_emit_probs[0, :]

    for t in range(1, seq_length):
        current_base = sequence[t].upper()
        current_emit_penalty = log_emit_probs[t]
        # transition_matrix = init_transition_matrix.copy()
        if current_base == 'A':
            transition_matrix = transition_matrix_A
        elif current_base == 'G':
            transition_matrix = transition_matrix_G
        elif current_base == 'C':
            transition_matrix = transition_matrix_C
        elif current_base == 'T':
            transition_matrix = transition_matrix_T
        else:
            transition_matrix = transition_matrix_G

        transition_matrix = set_transition_matrix(transition_matrix, states_to_num, intron_group, min_intron_length, min_intron_length_rare, current_base,
                                                  exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty, intron_quit_penalty, threshold, current_emit_penalty)

        current_penalty = transition_matrix + log_emit_probs[t]
        total_probs = dp[t - 1, :, None] + current_penalty
        dp[t, :] = np.max(total_probs, axis=0)
        path[t, :] = np.argmax(total_probs, axis=0)

    # best_path = [np.argmax(dp[-1, :])]
    best_path = [0]
    for t in range(seq_length - 1, 0, -1):
        best_path.append(path[t, best_path[-1]])
    best_path.reverse()

    return [states[i] for i in best_path]
