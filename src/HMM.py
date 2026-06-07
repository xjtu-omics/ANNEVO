import numpy as np
from tqdm import tqdm
from numba import njit


def define_state(min_intron_length):
    states = [
        'intergenic',
        'start0', 'start1', 'start2',
        'CDS0', 'CDS0_T', 'CDS1', 'CDS1_TA', 'CDS1_TG', 'CDS2',
        'DSS0', 'DSS1', 'DSS2', 'DSS0_T', 'DSS1_TA', 'DSS1_TG',
        'ASS0', 'ASS1', 'ASS2', 'ASS0_T', 'ASS1_TA', 'ASS1_TG',
        'end0', 'end1_TA', 'end1_TG', 'end2',
        'intron0_splice0', 'intron0_splice1', 'intron0_splice2', 'intron0_splice3',
        'intron0_T_splice0', 'intron0_T_splice1', 'intron0_T_splice2', 'intron0_T_splice3',
        'intron1_splice0', 'intron1_splice1', 'intron1_splice2', 'intron1_splice3',
        'intron1_TA_splice0', 'intron1_TA_splice1', 'intron1_TA_splice2', 'intron1_TA_splice3',
        'intron1_TG_splice0', 'intron1_TG_splice1', 'intron1_TG_splice2', 'intron1_TG_splice3',
        'intron2_splice0', 'intron2_splice1', 'intron2_splice2', 'intron2_splice3',
    ]

    for i in range(min_intron_length):
        states.append(f'intron0_{i}')
        states.append(f'intron0_T_{i}')
        states.append(f'intron1_{i}')
        states.append(f'intron1_TA_{i}')
        states.append(f'intron1_TG_{i}')
        states.append(f'intron2_{i}')

    states_to_num = {s: i for i, s in enumerate(states)}
    num_states = len(states_to_num)
    return states_to_num, num_states


def define_columns(states_to_num):
    column_groups = {
        'INTERGENIC': ['intergenic'],
        'CODING_EXON_0': ['CDS0', 'CDS0_T'],
        'CODING_EXON_1': ['CDS1', 'CDS1_TA', 'CDS1_TG'],
        'CODING_EXON_2': ['CDS2'],
        'INTRON_0': [state for state in states_to_num if state.startswith('intron0_')],
        'INTRON_1': [state for state in states_to_num if state.startswith('intron1_')],
        'INTRON_2': [state for state in states_to_num if state.startswith('intron2_')],
        'DSS_0': ['DSS0', 'DSS0_T'],
        'DSS_1': ['DSS1', 'DSS1_TA', 'DSS1_TG'],
        'DSS_2': ['DSS2'],
        'ASS_0': ['ASS0', 'ASS0_T'],
        'ASS_1': ['ASS1', 'ASS1_TA', 'ASS1_TG'],
        'ASS_2': ['ASS2'],
        'START': ['start0', 'start1', 'start2'],
        'END': ['end0', 'end1_TA', 'end1_TG', 'end2'],
    }

    column_dict = {}
    for column_name, state_names in column_groups.items():
        column_dict[column_name] = [states_to_num[state] for state in state_names]

    return column_dict


def set_transition_matrix_conditional_state(init_transition_matrix, states_to_num, min_intron_length,
                                            exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty,
                                            intron_quit_penalty):
    transition_matrix_A = init_transition_matrix.copy()
    transition_matrix_G = init_transition_matrix.copy()
    transition_matrix_C = init_transition_matrix.copy()
    transition_matrix_T = init_transition_matrix.copy()
    transition_matrix_other = init_transition_matrix.copy()

    # ---------------------------------- current base == A ----------------------------------
    # CDS-related
    transition_matrix_A[states_to_num[f'intergenic'], states_to_num[f'start0']] = 0
    transition_matrix_A[states_to_num[f'end0'], states_to_num[f'end1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'end1_TA'], states_to_num[f'end2']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'end1_TG'], states_to_num[f'end2']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num[f'start2'], states_to_num[f'DSS0']] = exon_sustain_penalty

    transition_matrix_A[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['CDS2'], states_to_num['DSS0']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS2'], states_to_num['DSS0']] = exon_sustain_penalty

    transition_matrix_A[states_to_num['CDS0_T'], states_to_num['CDS1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['CDS0_T'], states_to_num['DSS1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS0_T'], states_to_num['CDS1_TA']] = exon_sustain_penalty
    transition_matrix_A[states_to_num['ASS0_T'], states_to_num['DSS1_TA']] = exon_sustain_penalty

    # intron-related
    transition_matrix_A[states_to_num[f'intron0_{min_intron_length - 1}'], states_to_num[f'intron0_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron0_T_{min_intron_length - 1}'], states_to_num[f'intron0_T_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron1_{min_intron_length - 1}'], states_to_num[f'intron1_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron1_TA_{min_intron_length - 1}'], states_to_num[f'intron1_TA_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron1_TG_{min_intron_length - 1}'], states_to_num[f'intron1_TG_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron2_{min_intron_length - 1}'], states_to_num[f'intron2_splice2']] = intron_sustain_penalty
    transition_matrix_A[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_A[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1_TA']] = intron_quit_penalty
    transition_matrix_A[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_A[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == T ----------------------------------
    # CDS-related
    transition_matrix_T[states_to_num[f'start0'], states_to_num[f'start1']] = 0
    transition_matrix_T[states_to_num[f'start2'], states_to_num[f'CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num[f'start2'], states_to_num[f'DSS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num[f'CDS2'], states_to_num[f'end0']] = exon_sustain_penalty
    transition_matrix_T[states_to_num[f'ASS2'], states_to_num[f'end0']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS2'], states_to_num['CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS2'], states_to_num['DSS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS2'], states_to_num['CDS0_T']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS2'], states_to_num['DSS0_T']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS0_T'], states_to_num['DSS1']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS0_T'], states_to_num['DSS1']] = exon_sustain_penalty

    transition_matrix_T[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['CDS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_T[states_to_num['ASS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty

    # intron_related
    transition_matrix_T[states_to_num['intron0_splice0'], states_to_num['intron0_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron0_T_splice0'], states_to_num['intron0_T_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron1_splice0'], states_to_num['intron1_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron1_TG_splice0'], states_to_num['intron1_TG_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron1_TA_splice0'], states_to_num['intron1_TA_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num['intron2_splice0'], states_to_num['intron2_splice1']] = intron_sustain_penalty
    transition_matrix_T[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron1_TA_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron1_TG_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0_T']] = intron_quit_penalty
    transition_matrix_T[states_to_num[f'intron2_splice3'], states_to_num[f'end0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == G ----------------------------------
    # CDS-related
    transition_matrix_G[states_to_num[f'start1'], states_to_num[f'start2']] = 0
    transition_matrix_G[states_to_num[f'end0'], states_to_num[f'end1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num[f'end1_TA'], states_to_num[f'end2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num[f'start2'], states_to_num[f'DSS0']] = exon_sustain_penalty

    transition_matrix_G[states_to_num['CDS0_T'], states_to_num['CDS1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS0_T'], states_to_num['CDS1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS0_T'], states_to_num['DSS1_TG']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS0_T'], states_to_num['DSS1_TG']] = exon_sustain_penalty

    transition_matrix_G[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty

    transition_matrix_G[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['CDS2'], states_to_num['DSS0']] = exon_sustain_penalty
    transition_matrix_G[states_to_num['ASS2'], states_to_num['DSS0']] = exon_sustain_penalty

    # intron_related
    transition_matrix_G[states_to_num['DSS0'], states_to_num['intron0_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS0_T'], states_to_num['intron0_T_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS1'], states_to_num['intron1_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS1_TG'], states_to_num['intron1_TG_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS1_TA'], states_to_num['intron1_TA_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['DSS2'], states_to_num['intron2_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['start2'], states_to_num['intron2_splice0']] = exon_quit_penalty
    transition_matrix_G[states_to_num['intron0_splice2'], states_to_num['intron0_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron0_T_splice2'], states_to_num['intron0_T_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron1_splice2'], states_to_num['intron1_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron1_TG_splice2'], states_to_num['intron1_TG_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron1_TA_splice2'], states_to_num['intron1_TA_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num['intron2_splice2'], states_to_num['intron2_splice3']] = intron_sustain_penalty
    transition_matrix_G[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1_TG']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron1_TG_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_G[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == C ----------------------------------
    # CDS-related
    transition_matrix_C[states_to_num[f'start2'], states_to_num[f'CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num[f'start2'], states_to_num[f'DSS0']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TG'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TG'], states_to_num['DSS2']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS2'], states_to_num['DSS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS2'], states_to_num['CDS0']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS2'], states_to_num['DSS0']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS0_T'], states_to_num['DSS1']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS0_T'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS0_T'], states_to_num['DSS1']] = exon_sustain_penalty

    transition_matrix_C[states_to_num['CDS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TA'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['CDS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix_C[states_to_num['ASS1_TA'], states_to_num['DSS2']] = exon_sustain_penalty

    # intron-related
    transition_matrix_C[states_to_num['intron0_splice0'], states_to_num['intron0_splice1']] = intron_sustain_penalty - 10
    transition_matrix_C[states_to_num['intron0_T_splice0'], states_to_num['intron0_T_splice1']] = intron_sustain_penalty - 10
    transition_matrix_C[states_to_num['intron1_splice0'], states_to_num['intron1_splice1']] = intron_sustain_penalty - 10
    transition_matrix_C[states_to_num['intron1_TG_splice0'], states_to_num['intron1_TG_splice1']] = intron_sustain_penalty - 10
    transition_matrix_C[states_to_num['intron1_TA_splice0'], states_to_num['intron1_TA_splice1']] = intron_sustain_penalty - 10
    transition_matrix_C[states_to_num['intron2_splice0'], states_to_num['intron2_splice1']] = intron_sustain_penalty - 10

    transition_matrix_C[states_to_num[f'intron0_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron0_T_splice3'], states_to_num[f'ASS1']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron1_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron1_TA_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron1_TG_splice3'], states_to_num[f'ASS2']] = intron_quit_penalty
    transition_matrix_C[states_to_num[f'intron2_splice3'], states_to_num[f'ASS0']] = intron_quit_penalty
    # ---------------------------------------------------------------------------------------

    # ---------------------------------- current base == N ----------------------------------
    # Open any ATCG-available conditional path under 'N/other' with a fixed penalty.
    atcg_open_mask = (
        np.isfinite(transition_matrix_A) |
        np.isfinite(transition_matrix_T) |
        np.isfinite(transition_matrix_G) |
        np.isfinite(transition_matrix_C)
    )
    conditional_open_mask = atcg_open_mask & (~np.isfinite(init_transition_matrix))
    transition_matrix_other[conditional_open_mask] = -10
    # ---------------------------------------------------------------------------------------

    transition_matrix_dict = {
        'A': transition_matrix_A,
        'T': transition_matrix_T,
        'C': transition_matrix_C,
        'G': transition_matrix_G,
        'N': transition_matrix_other,
    }
    return transition_matrix_dict


def set_transition_matrix_common_state(transition_matrix, states_to_num, min_intron_length,
                                       exon_sustain_penalty, exon_quit_penalty, intron_sustain_penalty,
                                       intron_quit_penalty):
    transition_matrix[states_to_num['intergenic'], states_to_num['intergenic']] = 0
    transition_matrix[states_to_num['end2'], states_to_num['intergenic']] = exon_quit_penalty

    transition_matrix[states_to_num['CDS0'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS0'], states_to_num['CDS1']] = exon_sustain_penalty
    transition_matrix[states_to_num['CDS0'], states_to_num['DSS1']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS0'], states_to_num['DSS1']] = exon_sustain_penalty

    transition_matrix[states_to_num['CDS1'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS1'], states_to_num['CDS2']] = exon_sustain_penalty
    transition_matrix[states_to_num['ASS1'], states_to_num['DSS2']] = exon_sustain_penalty
    transition_matrix[states_to_num['CDS1'], states_to_num['DSS2']] = exon_sustain_penalty

    # Bridge donor motif helper states back to intron bodies.
    transition_matrix[states_to_num['intron0_splice1'], states_to_num['intron0_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron0_T_splice1'], states_to_num['intron0_T_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron1_splice1'], states_to_num['intron1_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron1_TA_splice1'], states_to_num['intron1_TA_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron1_TG_splice1'], states_to_num['intron1_TG_0']] = intron_sustain_penalty
    transition_matrix[states_to_num['intron2_splice1'], states_to_num['intron2_0']] = intron_sustain_penalty

    transition_matrix[states_to_num[f'intron0_0'], states_to_num[f'intron0_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron0_T_0'], states_to_num[f'intron0_T_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron1_0'], states_to_num[f'intron1_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron1_TA_0'], states_to_num[f'intron1_TA_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron1_TG_0'], states_to_num[f'intron1_TG_0']] = intron_sustain_penalty
    transition_matrix[states_to_num[f'intron2_0'], states_to_num[f'intron2_0']] = intron_sustain_penalty
    if min_intron_length > 1:
        for i in range(min_intron_length - 1):
            transition_matrix[states_to_num[f'intron0_{i}'], states_to_num[f'intron0_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron0_T_{i}'], states_to_num[f'intron0_T_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron1_{i}'], states_to_num[f'intron1_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron1_TA_{i}'], states_to_num[f'intron1_TA_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron1_TG_{i}'], states_to_num[f'intron1_TG_{i + 1}']] = intron_sustain_penalty
            transition_matrix[states_to_num[f'intron2_{i}'], states_to_num[f'intron2_{i + 1}']] = intron_sustain_penalty

    return transition_matrix


@njit(cache=True)
def _viterbi_core_numba_float32(log_emit_probs, transition_matrices, sequence_codes):
    seq_length, num_states = log_emit_probs.shape
    path = np.zeros((seq_length, num_states), dtype=np.int32)
    dp = np.full((seq_length, num_states), -np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for t in range(1, seq_length):
        transition_matrix = transition_matrices[sequence_codes[t]]
        for to_state in range(num_states):
            best_score = -np.inf
            best_from = 0
            emit_score = log_emit_probs[t, to_state]
            for from_state in range(num_states):
                score = dp[t - 1, from_state] + transition_matrix[from_state, to_state] + emit_score
                if score > best_score:
                    best_score = score
                    best_from = from_state
            dp[t, to_state] = best_score
            path[t, to_state] = best_from

    best_path = np.empty(seq_length, dtype=np.int32)
    best_path[seq_length - 1] = 0
    for t in range(seq_length - 1, 0, -1):
        best_path[t - 1] = path[t, best_path[t]]
    return best_path


@njit(cache=True)
def _viterbi_core_numba_float64(log_emit_probs, transition_matrices, sequence_codes):
    seq_length, num_states = log_emit_probs.shape
    path = np.zeros((seq_length, num_states), dtype=np.int32)
    dp = np.full((seq_length, num_states), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for t in range(1, seq_length):
        transition_matrix = transition_matrices[sequence_codes[t]]
        for to_state in range(num_states):
            best_score = -np.inf
            best_from = 0
            emit_score = log_emit_probs[t, to_state]
            for from_state in range(num_states):
                score = dp[t - 1, from_state] + transition_matrix[from_state, to_state] + emit_score
                if score > best_score:
                    best_score = score
                    best_from = from_state
            dp[t, to_state] = best_score
            path[t, to_state] = best_from

    best_path = np.empty(seq_length, dtype=np.int32)
    best_path[seq_length - 1] = 0
    for t in range(seq_length - 1, 0, -1):
        best_path[t - 1] = path[t, best_path[t]]
    return best_path


def _encode_sequence(sequence):
    base_to_code = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    return np.array([base_to_code.get(base, 4) for base in sequence], dtype=np.int32)


def viterbi_decoding(predictions, sequence, states_to_num, num_states, columns_dict, min_intron_length,
                     expect_exon=None, expect_intron=None, extra_penalty=None):
    """
    Decoding gene structure using viterbi algorithm.
    """
    np.seterr(divide='ignore', invalid='ignore')
    epsilon = 1e-3
    predictions[predictions < epsilon] = epsilon
    seq_length = predictions.shape[0]
    log_emit_probs = np.zeros((seq_length, num_states), dtype=np.float32)

    log_emit_probs[:, columns_dict['INTERGENIC']] = np.log(predictions[:, 0][:, np.newaxis])
    log_emit_probs[:, columns_dict['CODING_EXON_0']] = np.log(predictions[:, 1][:, np.newaxis])
    log_emit_probs[:, columns_dict['CODING_EXON_1']] = np.log(predictions[:, 3][:, np.newaxis])
    log_emit_probs[:, columns_dict['CODING_EXON_2']] = np.log(predictions[:, 2][:, np.newaxis])
    log_emit_probs[:, columns_dict['INTRON_0']] = np.log(predictions[:, 4][:, np.newaxis])
    log_emit_probs[:, columns_dict['INTRON_1']] = np.log(predictions[:, 6][:, np.newaxis])
    log_emit_probs[:, columns_dict['INTRON_2']] = np.log(predictions[:, 5][:, np.newaxis])
    log_emit_probs[:, columns_dict['DSS_0']] = np.log(predictions[:, 7][:, np.newaxis])
    log_emit_probs[:, columns_dict['DSS_1']] = np.log(predictions[:, 9][:, np.newaxis])
    log_emit_probs[:, columns_dict['DSS_2']] = np.log(predictions[:, 8][:, np.newaxis])
    log_emit_probs[:, columns_dict['ASS_0']] = np.log(predictions[:, 10][:, np.newaxis])
    log_emit_probs[:, columns_dict['ASS_1']] = np.log(predictions[:, 12][:, np.newaxis])
    log_emit_probs[:, columns_dict['ASS_2']] = np.log(predictions[:, 11][:, np.newaxis])
    log_emit_probs[:, columns_dict['START']] = np.log(predictions[:, 13][:, np.newaxis])
    log_emit_probs[:, columns_dict['END']] = np.log(predictions[:, 14][:, np.newaxis])

    # coding_prediction_columns = [1, 3, 2, 7, 9, 8, 10, 12, 11, 13, 14]
    # coding_columns_sum = predictions[:, coding_prediction_columns].sum(axis=1)
    # low_coding_indices = np.where(coding_columns_sum < 0.1)[0]
    # if low_coding_indices.size > 0:
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['CODING_EXON_0'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['CODING_EXON_1'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['CODING_EXON_2'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['DSS_0'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['DSS_1'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['DSS_2'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['ASS_0'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['ASS_1'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['ASS_2'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['START'])] *= 10
    #     log_emit_probs[np.ix_(low_coding_indices, columns_dict['END'])] *= 10

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

    init_transition_matrix = np.full((num_states, num_states), -np.inf, dtype=np.float32)
    init_transition_matrix = set_transition_matrix_common_state(init_transition_matrix, states_to_num,
                                                                min_intron_length,
                                                                exon_sustain_penalty, exon_quit_penalty,
                                                                intron_sustain_penalty, intron_quit_penalty)
    transition_matrix_dict = set_transition_matrix_conditional_state(init_transition_matrix,
                                                                     states_to_num, min_intron_length,
                                                                     exon_sustain_penalty,
                                                                     exon_quit_penalty,
                                                                     intron_sustain_penalty,
                                                                     intron_quit_penalty)

    transition_matrices = np.stack([
        transition_matrix_dict['A'],
        transition_matrix_dict['T'],
        transition_matrix_dict['C'],
        transition_matrix_dict['G'],
        transition_matrix_dict['N'],
    ]).astype(np.float32)
    sequence_codes = _encode_sequence(sequence)
    if seq_length <= 1_000_000:
        best_path = _viterbi_core_numba_float32(
            log_emit_probs,
            transition_matrices,
            sequence_codes,
        )
    else:
        best_path = _viterbi_core_numba_float64(
            log_emit_probs,
            transition_matrices,
            sequence_codes,
        )
    return best_path.tolist()
