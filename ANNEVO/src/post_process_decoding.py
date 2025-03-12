import numpy as np
from Bio import SeqIO
from BCBio import GFF
import h5py
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from collections import defaultdict
import pandas as pd


def reverse_complement(dna_sequence):
    complement_map = str.maketrans('ATGCRMYWKBSHDVNXatgcrmywkbshdvnx', 'TACGRMYWKBSHDVNXtacgrmywkbshdvnx')
    return dna_sequence.translate(complement_map)[::-1]


def detect_gene_location(base_predictions, seq_length, min_threshold, max_threshold):
    """
    Detect the range of potential genes based on model's predictions.
    """
    genic_proba = base_predictions[:, 1:].astype(np.float64).sum(axis=1)
    step = 50
    genic_region = []
    genic_region_start = 0
    in_genic_region = False
    cumulative_sum = np.cumsum(genic_proba, dtype=np.float64)

    for start in range(0, seq_length, step):
        end = min(start + step, seq_length)
        if start == 0:
            windows_genic_base_mean = cumulative_sum[end - 1] / end
        else:
            windows_genic_base_mean = (cumulative_sum[end - 1] - cumulative_sum[start - 1]) / (end - start)

        if windows_genic_base_mean >= min_threshold:
            in_genic_region = True
        else:
            if in_genic_region:
                potential_genic_range = genic_proba[genic_region_start:end]
                count_above_threshold = np.sum(potential_genic_range > max_threshold)
                if count_above_threshold >= 60:
                    genic_region.append((genic_region_start, end))
                # above_threshold = np.any(potential_genic_range > max_threshold)
                # if above_threshold:
                #     genic_region.append((genic_region_start, end))
                in_genic_region = False
            genic_region_start = start + step
    if in_genic_region:
        genic_region.append((genic_region_start, seq_length))
    return genic_region


def viterbi_decoding(emit_prob, trans_prob, phase_prob, sequence):
    """
    Decoding gene structure using viterbi algorithm.
    """
    old_settings = np.seterr(divide='ignore')
    epsilon = 1e-64
    extra_penalty_coefficient = 5
    emit_prob[emit_prob < epsilon] = epsilon
    trans_prob[trans_prob < epsilon] = epsilon
    phase_prob[phase_prob < epsilon] = epsilon
    states_to_num = {
        'intergenic': 0,
        'five_UTR': 1,
        'five_UTR_GC-AG': 2,
        'five_UTR_GT-AG': 3,
        'five_UTR_AT-AC': 4,
        'start0': 5,
        'start0_GC-AG': 6,
        'start0_GT-AG': 7,
        'start0_AT-AC': 8,
        'start1': 9,
        'start1_GC-AG': 10,
        'start1_GT-AG': 11,
        'start1_AT-AC': 12,
        'start2': 13,
        'CDS0': 14,
        'CDS0_GC-AG': 15,
        'CDS0_GT-AG': 16,
        'CDS0_AT-AC': 17,
        'CDS1': 18,
        'CDS1_GC-AG': 19,
        'CDS1_GT-AG': 20,
        'CDS1_AT-AC': 21,
        'CDS2': 22,
        'CDS2_GC-AG': 23,
        'CDS2_GT-AG': 24,
        'CDS2_AT-AC': 25,
        'end0': 26,
        'end0_GC-AG': 27,
        'end0_GT-AG': 28,
        'end0_AT-AC': 29,
        'end1': 30,
        'end1_GC-AG': 31,
        'end1_GT-AG': 32,
        'end1_AT-AC': 33,
        'end2': 34,
        'end2_GC-AG': 35,
        'end2_GT-AG': 36,
        'end2_AT-AC': 37,
        'end3': 38,
        'three_UTR': 39,
        'three_UTR_GC-AG': 40,
        'three_UTR_GT-AG': 41,
        'three_UTR_AT-AC': 42,
    }
    num_states = len(states_to_num)
    states = list(range(num_states))
    seq_length = emit_prob.shape[0]
    new_arr = np.zeros((seq_length, num_states))
    CDS_phase_prob = 0.5 * phase_prob[:, 0:3] + 0.5 * emit_prob[:, 2][:, np.newaxis]
    all_columns = set(range(num_states))
    none_intron_columns = {0, 1, 39, 5, 14, 26, 9, 18, 30, 34, 13, 22, 38}
    intron_columns = list(all_columns - none_intron_columns)
    exon_columns = [1, 39, 5, 14, 26, 9, 18, 30, 34, 13, 22, 38]
    new_arr[:, 0] = np.log(emit_prob[:, 0]) * 1
    new_arr[:, [1, 39]] = np.log(emit_prob[:, 1][:, np.newaxis]) * 1
    new_arr[:, [5, 14, 26]] = np.log(CDS_phase_prob[:, 0][:, np.newaxis])
    new_arr[:, [9, 18, 30, 34]] = np.log(CDS_phase_prob[:, 2][:, np.newaxis])
    new_arr[:, [13, 22, 38]] = np.log(CDS_phase_prob[:, 1][:, np.newaxis])
    new_arr[:, intron_columns] = np.log(emit_prob[:, 3][:, np.newaxis])
    # repeat_index = [i for i, char in enumerate(sequence) if char.islower()]
    # columns = [1, 39, 5, 14, 26, 9, 18, 30, 34, 13, 22, 38]
    # for col in columns:
    #     new_arr[repeat_index, col] *= extra_penalty_coefficient

    log_emit_probs = new_arr
    log_trans_probs = np.log(trans_prob)
    penalty_epsilon = np.log(epsilon)
    path = np.zeros((seq_length, num_states), dtype=int)  # 路径矩阵，每个位置记录来自于前一位置的路径号
    dp = np.full((seq_length, num_states), -np.inf)
    dp[0, 0] = 0
    # dp[0, :] = log_emit_probs[0, :]
    init_transition_matrix = np.full((num_states, num_states), penalty_epsilon)
    np.seterr(**old_settings)

    for t in range(1, seq_length):
        state_unchanged = log_trans_probs[t - 1][0]
        state_changed = log_trans_probs[t - 1][1]
        # state_unchanged = 0
        # state_changed = 0
        transition_matrix = init_transition_matrix.copy()
        current_2base = sequence[t:t + 2].upper()
        pre_2base = sequence[t - 2:t].upper()
        current_base = sequence[t].upper()
        # define common transition state
        transition_matrix[states_to_num['intergenic'], states_to_num['five_UTR']] = state_changed
        transition_matrix[states_to_num['three_UTR'], states_to_num['intergenic']] = state_changed
        transition_matrix[states_to_num['start2'], states_to_num['CDS0']] = state_unchanged
        transition_matrix[states_to_num['CDS0'], states_to_num['CDS1']] = state_unchanged
        transition_matrix[states_to_num['CDS1'], states_to_num['CDS2']] = state_unchanged
        transition_matrix[states_to_num['CDS2'], states_to_num['CDS0']] = state_unchanged
        transition_matrix[states_to_num['end3'], states_to_num['three_UTR']] = state_changed
        # State_unchanged
        for state in states_to_num:
            if state not in ['start0', 'start1', 'start2', 'CDS0', 'CDS1', 'CDS2', 'end0', 'end1', 'end2', 'end3']:
                transition_matrix[states_to_num[state], states_to_num[state]] = state_unchanged
        if current_2base == 'GC':
            transition_matrix[states_to_num['five_UTR'], states_to_num['five_UTR_GC-AG']] = state_changed
            transition_matrix[states_to_num['start0'], states_to_num['start0_GC-AG']] = state_changed
            transition_matrix[states_to_num['start1'], states_to_num['start1_GC-AG']] = state_changed
            transition_matrix[states_to_num['start2'], states_to_num['CDS2_GC-AG']] = state_changed
            transition_matrix[states_to_num['CDS0'], states_to_num['CDS0_GC-AG']] = state_changed
            transition_matrix[states_to_num['CDS1'], states_to_num['CDS1_GC-AG']] = state_changed
            transition_matrix[states_to_num['CDS2'], states_to_num['CDS2_GC-AG']] = state_changed
            transition_matrix[states_to_num['end0'], states_to_num['end0_GC-AG']] = state_changed
            transition_matrix[states_to_num['end1'], states_to_num['end1_GC-AG']] = state_changed
            transition_matrix[states_to_num['end2'], states_to_num['end2_GC-AG']] = state_changed
            transition_matrix[states_to_num['end3'], states_to_num['three_UTR_GC-AG']] = state_changed
            transition_matrix[states_to_num['three_UTR'], states_to_num['three_UTR_GC-AG']] = state_changed
        if current_2base == 'GT':
            transition_matrix[states_to_num['five_UTR'], states_to_num['five_UTR_GT-AG']] = state_changed
            transition_matrix[states_to_num['start0'], states_to_num['start0_GT-AG']] = state_changed
            transition_matrix[states_to_num['start1'], states_to_num['start1_GT-AG']] = state_changed
            transition_matrix[states_to_num['start2'], states_to_num['CDS2_GT-AG']] = state_changed
            transition_matrix[states_to_num['CDS0'], states_to_num['CDS0_GT-AG']] = state_changed
            transition_matrix[states_to_num['CDS1'], states_to_num['CDS1_GT-AG']] = state_changed
            transition_matrix[states_to_num['CDS2'], states_to_num['CDS2_GT-AG']] = state_changed
            transition_matrix[states_to_num['end0'], states_to_num['end0_GT-AG']] = state_changed
            transition_matrix[states_to_num['end1'], states_to_num['end1_GT-AG']] = state_changed
            transition_matrix[states_to_num['end2'], states_to_num['end2_GT-AG']] = state_changed
            transition_matrix[states_to_num['end3'], states_to_num['three_UTR_GT-AG']] = state_changed
            transition_matrix[states_to_num['three_UTR'], states_to_num['three_UTR_GT-AG']] = state_changed
        if current_2base == 'AT':
            transition_matrix[states_to_num['five_UTR'], states_to_num['five_UTR_AT-AC']] = state_changed
            transition_matrix[states_to_num['start0'], states_to_num['start0_AT-AC']] = state_changed
            transition_matrix[states_to_num['start1'], states_to_num['start1_AT-AC']] = state_changed
            transition_matrix[states_to_num['start2'], states_to_num['CDS2_AT-AC']] = state_changed
            transition_matrix[states_to_num['CDS0'], states_to_num['CDS0_AT-AC']] = state_changed
            transition_matrix[states_to_num['CDS1'], states_to_num['CDS1_AT-AC']] = state_changed
            transition_matrix[states_to_num['CDS2'], states_to_num['CDS2_AT-AC']] = state_changed
            transition_matrix[states_to_num['end0'], states_to_num['end0_AT-AC']] = state_changed
            transition_matrix[states_to_num['end1'], states_to_num['end1_AT-AC']] = state_changed
            transition_matrix[states_to_num['end2'], states_to_num['end2_AT-AC']] = state_changed
            transition_matrix[states_to_num['end3'], states_to_num['three_UTR_AT-AC']] = state_changed
            transition_matrix[states_to_num['three_UTR'], states_to_num['three_UTR_AT-AC']] = state_changed
        if pre_2base == 'AG':
            if current_base == 'T':
                transition_matrix[states_to_num['start0_GC-AG'], states_to_num['start1']] = state_changed
                transition_matrix[states_to_num['start0_GT-AG'], states_to_num['start1']] = state_changed
                transition_matrix[states_to_num['CDS2_GC-AG'], states_to_num['end0']] = state_changed
                transition_matrix[states_to_num['CDS2_GT-AG'], states_to_num['end0']] = state_changed
            if current_base == 'G':
                transition_matrix[states_to_num['start1_GC-AG'], states_to_num['start2']] = state_changed
                transition_matrix[states_to_num['start1_GT-AG'], states_to_num['start2']] = state_changed
                transition_matrix[states_to_num['end0_GC-AG'], states_to_num['end1']] = state_changed
                transition_matrix[states_to_num['end0_GT-AG'], states_to_num['end1']] = state_changed
                transition_matrix[states_to_num['end2_GC-AG'], states_to_num['end3']] = state_changed
                transition_matrix[states_to_num['end2_GT-AG'], states_to_num['end3']] = state_changed
            if current_base == 'A':
                transition_matrix[states_to_num['five_UTR_GC-AG'], states_to_num['start0']] = state_changed
                transition_matrix[states_to_num['five_UTR_GT-AG'], states_to_num['start0']] = state_changed
                transition_matrix[states_to_num['end0_GC-AG'], states_to_num['end2']] = state_changed
                transition_matrix[states_to_num['end0_GT-AG'], states_to_num['end2']] = state_changed
                transition_matrix[states_to_num['end1_GC-AG'], states_to_num['end3']] = state_changed
                transition_matrix[states_to_num['end1_GT-AG'], states_to_num['end3']] = state_changed
                transition_matrix[states_to_num['end2_GC-AG'], states_to_num['end3']] = state_changed
                transition_matrix[states_to_num['end2_GT-AG'], states_to_num['end3']] = state_changed
            transition_matrix[states_to_num['five_UTR_GC-AG'], states_to_num['five_UTR']] = state_changed
            transition_matrix[states_to_num['five_UTR_GT-AG'], states_to_num['five_UTR']] = state_changed
            transition_matrix[states_to_num['CDS0_GC-AG'], states_to_num['CDS1']] = state_changed
            transition_matrix[states_to_num['CDS0_GT-AG'], states_to_num['CDS1']] = state_changed
            transition_matrix[states_to_num['CDS1_GC-AG'], states_to_num['CDS2']] = state_changed
            transition_matrix[states_to_num['CDS1_GT-AG'], states_to_num['CDS2']] = state_changed
            transition_matrix[states_to_num['CDS2_GC-AG'], states_to_num['CDS0']] = state_changed
            transition_matrix[states_to_num['CDS2_GT-AG'], states_to_num['CDS0']] = state_changed
            transition_matrix[states_to_num['three_UTR_GC-AG'], states_to_num['three_UTR']] = state_changed
            transition_matrix[states_to_num['three_UTR_GT-AG'], states_to_num['three_UTR']] = state_changed
        if pre_2base == 'AC':
            if current_base == 'T':
                transition_matrix[states_to_num['start0_AT-AC'], states_to_num['start1']] = state_changed
                transition_matrix[states_to_num['CDS2_AT-AC'], states_to_num['end0']] = state_changed
            if current_base == 'G':
                transition_matrix[states_to_num['start1_AT-AC'], states_to_num['start2']] = state_changed
                transition_matrix[states_to_num['end0_AT-AC'], states_to_num['end1']] = state_changed
                transition_matrix[states_to_num['end2_AT-AC'], states_to_num['end3']] = state_changed
            if current_base == 'A':
                transition_matrix[states_to_num['five_UTR_AT-AC'], states_to_num['start0']] = state_changed
                transition_matrix[states_to_num['end0_AT-AC'], states_to_num['end2']] = state_changed
                transition_matrix[states_to_num['end1_AT-AC'], states_to_num['end3']] = state_changed
                transition_matrix[states_to_num['end2_AT-AC'], states_to_num['end3']] = state_changed
            transition_matrix[states_to_num['five_UTR_AT-AC'], states_to_num['five_UTR']] = state_changed
            transition_matrix[states_to_num['CDS0_AT-AC'], states_to_num['CDS1']] = state_changed
            transition_matrix[states_to_num['CDS1_AT-AC'], states_to_num['CDS2']] = state_changed
            transition_matrix[states_to_num['CDS2_AT-AC'], states_to_num['CDS0']] = state_changed
            transition_matrix[states_to_num['three_UTR_AT-AC'], states_to_num['three_UTR']] = state_changed
        if current_base == 'A':
            transition_matrix[states_to_num['five_UTR'], states_to_num['start0']] = state_changed
            transition_matrix[states_to_num['end0'], states_to_num['end2']] = state_unchanged
            transition_matrix[states_to_num['end1'], states_to_num['end3']] = state_unchanged
            transition_matrix[states_to_num['end2'], states_to_num['end3']] = state_unchanged
        if current_base == 'T':
            transition_matrix[states_to_num['start0'], states_to_num['start1']] = state_unchanged
            transition_matrix[states_to_num['CDS2'], states_to_num['end0']] = state_unchanged
        if current_base == 'G':
            transition_matrix[states_to_num['start1'], states_to_num['start2']] = state_unchanged
            transition_matrix[states_to_num['end0'], states_to_num['end1']] = state_unchanged
            transition_matrix[states_to_num['end2'], states_to_num['end3']] = state_unchanged

        current_penalty = transition_matrix + log_emit_probs[t]
        current_penalty[:, states_to_num['start0']] = current_penalty[:, states_to_num['start0']] * extra_penalty_coefficient
        current_penalty[:, states_to_num['start1']] = current_penalty[:, states_to_num['start1']] * extra_penalty_coefficient
        current_penalty[:, states_to_num['start2']] = current_penalty[:, states_to_num['start2']] * extra_penalty_coefficient
        current_penalty[:, states_to_num['end0']] = current_penalty[:, states_to_num['end0']] * extra_penalty_coefficient
        current_penalty[:, states_to_num['end1']] = current_penalty[:, states_to_num['end1']] * extra_penalty_coefficient
        current_penalty[:, states_to_num['end2']] = current_penalty[:, states_to_num['end2']] * extra_penalty_coefficient
        current_penalty[:, states_to_num['end3']] = current_penalty[:, states_to_num['end3']] * extra_penalty_coefficient
        total_probs = dp[t - 1, :, None] + current_penalty
        dp[t, :] = np.max(total_probs, axis=0)
        path[t, :] = np.argmax(total_probs, axis=0)

    # best_path = [np.argmax(dp[-1, :])]
    best_path = [0]
    for t in range(seq_length - 1, 0, -1):
        best_path.append(path[t, best_path[-1]])
    best_path.reverse()
    return [states[i] for i in best_path], CDS_phase_prob


def calculate_gene_score(gene_structure, CDS_prob, intron_prob, CDS_list, intron_list, sequence):
    CDS_phase0 = [5, 14, 26]
    CDS_phase2 = [9, 18, 30, 34]
    CDS_phase1 = [13, 22, 38]
    CDS_score = 0
    intron_score = 0
    CDS_num = 0
    intron_num = 0

    CDS_score_list = []
    intron_score_list = []
    for CDS_start, CDS_end in CDS_list:
        CDS_score_single_sum = 0
        CDS_num_single = CDS_end - CDS_start + 1
        coefficient = 0
        for i in range(CDS_start, CDS_end + 1):
            if gene_structure[i] in CDS_phase0:
                if sequence[i].islower():
                    CDS_score += CDS_prob[i, 0] * coefficient
                    CDS_score_single_sum += CDS_prob[i, 0] * coefficient
                else:
                    CDS_score += CDS_prob[i, 0]
                    CDS_score_single_sum += CDS_prob[i, 0]
                CDS_num += 1
            elif gene_structure[i] in CDS_phase2:
                if sequence[i].islower():
                    CDS_score += CDS_prob[i, 2] * coefficient
                    CDS_score_single_sum += CDS_prob[i, 2] * coefficient
                else:
                    CDS_score += CDS_prob[i, 2]
                    CDS_score_single_sum += CDS_prob[i, 2]
                CDS_num += 1
            elif gene_structure[i] in CDS_phase1:
                if sequence[i].islower():
                    CDS_score += CDS_prob[i, 1] * coefficient
                    CDS_score_single_sum += CDS_prob[i, 1] * coefficient
                else:
                    CDS_score += CDS_prob[i, 1]
                    CDS_score_single_sum += CDS_prob[i, 1]
                CDS_num += 1
        CDS_score_single = CDS_score_single_sum / CDS_num_single
        CDS_score_list.append(CDS_score_single)
    # for intron_start, intron_end in intron_list:
    #     intron_score += intron_prob[intron_start: intron_end + 1].sum()
    #     intron_num += (intron_end - intron_start + 1)
    #     intron_score_single = intron_prob[intron_start: intron_end + 1].sum() / (intron_end - intron_start + 1)
    #     intron_score_list.append(intron_score_single)
    CDS_score = CDS_score / CDS_num if CDS_num != 0 else 0
    intron_score = intron_score / intron_num if intron_num != 0 else 0
    return CDS_score, intron_score, CDS_score_list, intron_score_list


def parse_ranges(lst, targets):
    def process_sublist(sublist, start_index, target_values):
        sub_ranges = {'exon': [], 'CDS': [], 'intron': []}
        df = pd.DataFrame({'value': sublist})
        for target in target_values:
            if isinstance(target, tuple):
                mask = df['value'].isin(target)
                key = 'exon'
            elif target == 2:
                mask = df['value'] == target
                key = 'CDS'
            elif target == 3:
                mask = df['value'] == target
                key = 'intron'
            else:
                continue

            if mask.any():
                df_target = df[mask]
                groups = (df_target.index.to_series().diff() != 1).cumsum()
                grouped = df_target.groupby(groups).apply(lambda x: (x.index.min(), x.index.max()))
                sub_ranges[key].extend([(start + start_index, end + start_index) for start, end in grouped.tolist()])
        return sub_ranges

    sublists = []
    start_idx = 0
    in_zero_sequence = False
    for i, value in enumerate(lst):
        if value == 0:
            if not in_zero_sequence:
                if i != start_idx:
                    sublists.append((lst[start_idx:i], start_idx))
                in_zero_sequence = True
                start_idx = i + 1
        else:
            in_zero_sequence = False

    if start_idx < len(lst) and not all(v == 0 for v in lst[start_idx:]):
        sublists.append((lst[start_idx:], start_idx))

    all_sublist_ranges = []
    for sublist, start_index in sublists:
        sub_ranges = process_sublist(sublist, start_index, targets)
        all_sublist_ranges.append(sub_ranges)

    return all_sublist_ranges


def decode_gene_structure(location_start, base_predictions, transition_predictions, phase_predictions, sequence, min_cds_length, min_cds_score, single_cds_score):
    """
    1. Detect the possible location of the gene based on base_predictions;
    2. Decoding gene structure using viterbi algorithm based on base_predictions, phase_predictions and transition_predictions;
    3. Record the position of different classes.
    """
    all_columns = set(range(43))
    none_intron_columns = {0, 1, 39, 5, 14, 26, 9, 18, 30, 34, 13, 22, 38}
    intron_columns = all_columns - none_intron_columns
    gene_structure_all_states, CDS_prob = viterbi_decoding(base_predictions, transition_predictions, phase_predictions, sequence)
    intron_prob = base_predictions[:, 3]
    gene_structure_four_states = [
        0 if x in {0} else
        1 if x in {1, 39} else
        2 if x in {5, 14, 26, 9, 18, 30, 34, 13, 22, 38} else
        3 if x in intron_columns else x
        for x in gene_structure_all_states
    ]
    targets = [(1, 2), 2, 3]
    gene_list = parse_ranges(gene_structure_four_states, targets)
    filtered_gene_list = []
    for gene in gene_list:
        exon_list_init = gene['exon']
        CDS_list_init = gene['CDS']
        intron_list_init = gene['intron']

        if not CDS_list_init:
            continue
        CDS_count = sum((CDS[1] - CDS[0] + 1) for CDS in CDS_list_init)
        if CDS_count < min_cds_length:
            continue
        CDS_score, intron_score, CDS_score_list, intron_score_list = calculate_gene_score(gene_structure_all_states, CDS_prob, intron_prob, CDS_list_init, intron_list_init, sequence)
        if CDS_score < min_cds_score:
            continue
        exist_confident_cds = False
        for score in CDS_score_list:
            if score >= single_cds_score:
                exist_confident_cds = True
        if not exist_confident_cds:
            continue
        exon_list = [(start + location_start, end + location_start) for start, end in exon_list_init]
        CDS_list = [(start + location_start, end + location_start) for start, end in CDS_list_init]
        intron_list = [(start + location_start, end + location_start) for start, end in intron_list_init]
        first_CDS_position = CDS_list[0][0]
        gene_attribute = (exon_list, CDS_list, intron_list, CDS_score, intron_score, CDS_score_list, intron_score_list, first_CDS_position)
        filtered_gene_list.append(gene_attribute)
    return filtered_gene_list


def process_gene_segment(region, min_cds_length, min_cds_score, single_cds_score):
    location_start, location_end, seq_id, strand, base_slice, transition_slice, phase_slice, sequence_slice = region
    if location_start is None:
        gene_list = []
    else:
        gene_list = decode_gene_structure(
            location_start,
            base_slice,
            transition_slice,
            phase_slice,
            sequence_slice,
            min_cds_length,
            min_cds_score,
            single_cds_score,
        )
    return gene_list, seq_id, strand


def write_result(file, num, seq_id, result, length, strand):
    exon_list, CDS_list, intron_list, CDS_score, intron_score, CDS_score_list, intron_score_list, _ = result
    file.write(f'# Start gene g{num + 1}\n')
    file.write(f'# The CDS score is {CDS_score}\n')
    file.write(f'# The intron score is {intron_score}\n')
    if strand == 1:
        gene_start, gene_end = exon_list[0][0], exon_list[-1][1]
        gene_start, gene_end = gene_start + 1, gene_end + 1  # 0-based to 1-based
        file.write(f'{seq_id}\tANNEVO\tgene\t{gene_start}\t{gene_end}\t.\t+\t.\tID={seq_id}-g{num + 1}\n')
        file.write(f'{seq_id}\tANNEVO\ttranscript\t{gene_start}\t{gene_end}\t.\t+\t.\tID={seq_id}-g{num + 1}.t1;Parent={seq_id}-g{num + 1}\n')
        for i, exon in enumerate(exon_list):
            start, end = exon
            type_start, type_end = start + 1, end + 1
            file.write(f'{seq_id}\tANNEVO\texon\t{type_start}\t{type_end}\t.\t+\t.\tID={seq_id}-g{num + 1}.t1.exon.{i + 1};Parent={seq_id}-g{num + 1}.t1\n')
        CDS_num = 0
        for i, CDS in enumerate(CDS_list):
            start, end = CDS
            type_start, type_end = start + 1, end + 1
            phase_map = [0, 2, 1]
            phase = phase_map[CDS_num]
            CDS_num = (CDS_num + type_end - type_start + 1) % 3
            CDS_score_single = CDS_score_list[i]
            file.write(f'{seq_id}\tANNEVO\tCDS\t{type_start}\t{type_end}\t{CDS_score_single:.2f}\t+\t{phase}\tID={seq_id}-g{num + 1}.t1.CDS.{i + 1};Parent={seq_id}-g{num + 1}.t1\n')
    else:
        gene_start = length - (exon_list[-1][1] + 1) + 1
        gene_end = length - (exon_list[0][0] + 1) + 1
        file.write(f'{seq_id}\tANNEVO\tgene\t{gene_start}\t{gene_end}\t.\t-\t.\tID={seq_id}-g{num + 1}\n')
        file.write(f'{seq_id}\tANNEVO\ttranscript\t{gene_start}\t{gene_end}\t.\t-\t.\tID={seq_id}-g{num + 1}.t1;Parent={seq_id}-g{num + 1}\n')
        CDS_num = 0
        for i, exon in enumerate(exon_list):
            start, end = exon
            type_start, type_end = length - (end + 1) + 1, length - (start + 1) + 1
            file.write(f'{seq_id}\tANNEVO\texon\t{type_start}\t{type_end}\t.\t-\t.\tID={seq_id}-g{num + 1}.t1.exon.{i + 1};Parent={seq_id}-g{num + 1}.t1\n')
        for i, CDS in enumerate(CDS_list):
            start, end = CDS
            type_start, type_end = length - (end + 1) + 1, length - (start + 1) + 1
            phase_map = [0, 2, 1]
            phase = phase_map[CDS_num]
            CDS_num = (CDS_num + type_end - type_start + 1) % 3
            CDS_score_single = CDS_score_list[i]
            file.write(f'{seq_id}\tANNEVO\tCDS\t{type_start}\t{type_end}\t{CDS_score_single:.2f}\t-\t{phase}\tID={seq_id}-g{num + 1}.t1.CDS.{i + 1};Parent={seq_id}-g{num + 1}.t1\n')

    file.write(f'# End gene g{num + 1}\n')
    file.write(f'###\n')


def predict_gff(genome, model_prediction_path, output, cpu_num, average_threshold, max_threshold, min_cds_length, min_cds_score, single_cds_score):
    with open(genome) as fna:
        genome_seq = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    with open(output, 'w') as file:
        file.write('# This output was generated with ANNEVO (version 1.0.0).\n')
        file.write('# ANNEVO is a gene prediction tool written by YeLab.\n')
    seq_num = 1
    prediction_files = [f for f in os.listdir(model_prediction_path) if os.path.isfile(os.path.join(model_prediction_path, f))]

    for prediction_file in prediction_files:
        genome_predictions = {}
        with h5py.File(f'{model_prediction_path}/{prediction_file}', 'r') as h5file:
            for chromosome in h5file.keys():
                data = []
                chr_group = h5file[chromosome]
                labels = ['base_predictions_forward', 'transition_predictions_forward', 'phase_predictions_forward',
                          'base_predictions_reverse', 'transition_predictions_reverse', 'phase_predictions_reverse']
                for label in labels:
                    dataset = np.array(chr_group[label])
                    data.append(dataset)
                genome_predictions[chromosome] = data
        potential_gene_list = []
        chromosome_length = {}

        for chromosome in genome_predictions:
            chromosome_seq_record = genome_seq[chromosome]
            sequence_forward = str(chromosome_seq_record.seq)
            sequence_reverse = reverse_complement(sequence_forward)
            length = len(sequence_forward)
            chromosome_length[chromosome] = length
            base_predictions_forward, transition_predictions_forward, phase_predictions_forward, \
                base_predictions_reverse, transition_predictions_reverse, phase_predictions_reverse = genome_predictions[chromosome]

            # position index conversion
            '''
            The position tuple in the forward array is (a, b) 
            The position tuple in the forward chains of gff is (a + 1, b)
            The position tuple in the reverse chains of gff is (length - b + 1, length - (a + 1) + 1) = (length - b + 1, length - a)
            The position tuple in the reverse array is (length - b, length - a) 
            '''

            potential_gene_chromosome_forward = detect_gene_location(base_predictions_forward, length, average_threshold, max_threshold)
            if not potential_gene_chromosome_forward:
                potential_gene_list.append(
                    (None, None, chromosome, 1, None, None, None, None)
                )
            else:
                for location_start, location_end in potential_gene_chromosome_forward:
                    potential_gene_list.append(
                        (location_start, location_end, chromosome, 1,
                         base_predictions_forward[location_start:location_end],
                         transition_predictions_forward[location_start:location_end],
                         phase_predictions_forward[location_start:location_end],
                         sequence_forward[location_start:location_end])
                    )
            potential_gene_chromosome_reverse = detect_gene_location(base_predictions_reverse, length, average_threshold, max_threshold)
            if not potential_gene_chromosome_reverse:
                potential_gene_list.append(
                    (None, None, chromosome, -1, None, None, None, None)
                )
            else:
                for location_start, location_end in potential_gene_chromosome_reverse:
                    potential_gene_list.append(
                        (location_start, location_end, chromosome, -1,
                         base_predictions_reverse[location_start:location_end],
                         transition_predictions_reverse[location_start:location_end],
                         phase_predictions_reverse[location_start:location_end],
                         sequence_reverse[location_start:location_end])
                    )

        results = []
        with ProcessPoolExecutor(max_workers=cpu_num) as executor:
            future_to_segment = {executor.submit(process_gene_segment, region, min_cds_length, min_cds_score, single_cds_score): region for region in potential_gene_list}
            for future in as_completed(future_to_segment):
                # try:
                result = future.result()
                results.append(result)
                # except Exception as e:
                #     print(f"Error processing {future_to_segment[future]}: {e}")

        grouped_results = defaultdict(list)
        for gene_set, seq_id, strand in results:
            decode_gene = (gene_set, strand)
            grouped_results[seq_id].append(decode_gene)

        with open(output, 'a') as file:
            for chromosome in grouped_results:
                gene_list_forward = []
                gene_list_reverse = []
                length = chromosome_length[chromosome]
                gene_num = 0
                for gene_set, strand in grouped_results[chromosome]:
                    if gene_set:
                        for gene in gene_set:
                            if strand == 1:
                                gene_list_forward.append(gene)
                            else:
                                gene_list_reverse.append(gene)

                gene_list_forward.sort(key=lambda x: x[-1])
                gene_list_reverse.sort(key=lambda x: x[-1], reverse=True)
                file.write('#\n')
                file.write(f'# ----- prediction on sequence number {seq_num} (length = {length}, name = {chromosome}) -----\n')
                file.write('#\n')
                file.write(f'# Predicted genes for sequence number {seq_num} on forward strands\n')
                if not gene_list_forward:
                    file.write(f'# None\n')
                    file.write(f'###\n')
                else:
                    for gene in gene_list_forward:
                        write_result(file, gene_num, chromosome, gene, length=length, strand=1)
                        gene_num += 1
                        file.flush()
                file.write('#\n')
                file.write(f'# Predicted genes for sequence number {seq_num} on reverse strands\n')
                if not gene_list_reverse:
                    file.write(f'# None\n')
                    file.write(f'###\n')
                else:
                    for gene in gene_list_reverse:
                        write_result(file, gene_num, chromosome, gene, length=length, strand=-1)
                        gene_num += 1
                        file.flush()

                seq_num += 1
