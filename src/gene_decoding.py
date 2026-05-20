import numpy as np
from Bio import SeqIO
import h5py
from src.predict_nucleotide import rev_complement
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
from src.HMM import viterbi_decoding, define_state, define_columns
import time
import re
from tqdm import tqdm
from itertools import islice


def detect_gene_location(base_predictions, seq_length, min_threshold, max_threshold, min_cds_length=50):
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

        # windows_count_above_threshold = np.sum(genic_proba[start:end] > max_threshold)
        # if windows_genic_base_mean >= min_threshold or windows_count_above_threshold >= 1:
        #     in_genic_region = True
        if windows_genic_base_mean >= min_threshold:
            in_genic_region = True
        else:
            if in_genic_region:
                potential_genic_range = genic_proba[genic_region_start:end]
                count_above_threshold = np.sum(potential_genic_range > max_threshold)
                if count_above_threshold >= min_cds_length:
                    genic_region.append((genic_region_start, end))
                in_genic_region = False
            genic_region_start = start + step

    if in_genic_region:
        potential_genic_range = genic_proba[genic_region_start:seq_length]
        count_above_threshold = np.sum(potential_genic_range > max_threshold)
        if count_above_threshold >= min_cds_length:
            genic_region.append((genic_region_start, seq_length))
    return genic_region


def expand_and_merge_regions(regions, seq_length, buffer_size=100):
    if not regions:
        return []

    expanded = []
    for start, end in regions:
        s = max(0, int(start) - buffer_size)
        e = min(int(seq_length), int(end) + buffer_size)
        if s < e:
            expanded.append((s, e))

    if not expanded:
        return []

    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for s, e in expanded[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def calculate_gene_score(gene_structure, predictions, CDS_list, columns_dict):
    state_to_prediction_column = {}
    column_to_prediction = {
        'INTERGENIC': 0,
        'CODING_EXON_0': 1,
        'CODING_EXON_1': 3,
        'CODING_EXON_2': 2,
        'INTRON_0': 4,
        'INTRON_1': 6,
        'INTRON_2': 5,
        'DSS_0': 7,
        'DSS_1': 9,
        'DSS_2': 8,
        'ASS_0': 10,
        'ASS_1': 12,
        'ASS_2': 11,
        'START': 13,
        'END': 14,
    }
    for column_name, prediction_column in column_to_prediction.items():
        for state in columns_dict[column_name]:
            state_to_prediction_column[state] = prediction_column

    CDS_score_list = []
    gene_cds_score_sum = 0
    gene_cds_base_num = 0

    for CDS_start, CDS_end in CDS_list:
        cds_score_sum = 0
        cds_base_num = 0
        for i in range(CDS_start, CDS_end + 1):
            state = gene_structure[i]
            if state not in state_to_prediction_column:
                continue
            base_score = predictions[i, state_to_prediction_column[state]]

            cds_score_sum += base_score
            cds_base_num += 1
            gene_cds_score_sum += base_score
            gene_cds_base_num += 1

        CDS_score_list.append(cds_score_sum / cds_base_num)

    CDS_score = gene_cds_score_sum / gene_cds_base_num
    return CDS_score, CDS_score_list


def parse_ranges(lst, targets):
    def process_sublist(sublist, start_index, target_values):
        sub_ranges = {'CDS': [], 'intron': []}
        df = pd.DataFrame({'value': sublist})
        for target in target_values:
            if target == 1:
                mask = df['value'] == target
                key = 'CDS'
            elif target == 2:
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


def decode_gene_structure(location_start, predictions, sequence, min_cds_length, min_cds_score, min_intron_length):
    def smooth_phase_group(column_indices, keep_weight=0.8):
        phase_probs = predictions[:, column_indices]
        phase_mean = phase_probs.sum(axis=1, keepdims=True) / float(len(column_indices))
        predictions[:, column_indices] = phase_probs * keep_weight + phase_mean * (1.0 - keep_weight)

    def run_decoding(current_min_intron_length):
        decode_start_time = time.time()
        states_to_num, num_states = define_state(min_intron_length=current_min_intron_length)
        columns_dict = define_columns(states_to_num)
        gene_structure_all_states = viterbi_decoding(
            predictions,
            sequence,
            states_to_num,
            num_states,
            columns_dict,
            min_intron_length=current_min_intron_length
        )
        phase_0_columns = columns_dict['CODING_EXON_0'] + columns_dict['ASS_0'] + columns_dict['DSS_0']
        phase_1_columns = columns_dict['CODING_EXON_1'] + columns_dict['ASS_1'] + columns_dict['DSS_1']
        phase_2_columns = columns_dict['CODING_EXON_2'] + columns_dict['ASS_2'] + columns_dict['DSS_2']
        CDS_columns = set(
            phase_0_columns +
            phase_1_columns +
            phase_2_columns +
            columns_dict['START'] +
            columns_dict['END']
        )
        intron_columns = set(
            columns_dict['INTRON_0'] +
            columns_dict['INTRON_1'] +
            columns_dict['INTRON_2']
        )
        gene_structure_three_states = [
            0 if x in {states_to_num['intergenic']} else
            1 if x in CDS_columns else
            2 if x in intron_columns else x
            for x in gene_structure_all_states
        ]
        # print(gene_structure_three_states)
        gene_list = parse_ranges(gene_structure_three_states, [1, 2])
        decode_runtime = time.time() - decode_start_time
        return gene_structure_all_states, columns_dict, gene_list, decode_runtime

    # smooth_phase_group([1, 3, 2])
    # smooth_phase_group([7, 9, 8])
    # smooth_phase_group([10, 12, 11])
    # smooth_phase_group([4, 6, 5])

    epsilon = 1e-3
    predictions[predictions < epsilon] = epsilon
    gene_structure_all_states, columns_dict, gene_list, first_decode_time = run_decoding(current_min_intron_length=1)
    second_decode_time = 0.0
    rerun_with_target_min_intron = False
    if min_intron_length > 1:
        has_short_intron = False
        for gene in gene_list:
            for intron_start, intron_end in gene['intron']:
                intron_length = intron_end - intron_start + 1
                if intron_length < min_intron_length:
                    has_short_intron = True
                    break
            if has_short_intron:
                break

        if has_short_intron:
            rerun_with_target_min_intron = True
            gene_structure_all_states, columns_dict, gene_list, second_decode_time = run_decoding(
                current_min_intron_length=min_intron_length
            )

    filtered_gene_list = []
    for gene in gene_list:
        CDS_list_init = gene['CDS']

        if not CDS_list_init:
            continue
        CDS_count = sum((CDS[1] - CDS[0] + 1) for CDS in CDS_list_init)
        CDS_score, CDS_score_list = calculate_gene_score(gene_structure_all_states, predictions, CDS_list_init,
                                                         columns_dict)
        score_threshold = min_cds_score * 1.5 if CDS_count < min_cds_length or len(CDS_list_init) == 1 else min_cds_score
        if CDS_score < score_threshold:
            continue

        CDS_list = [(start + location_start, end + location_start) for start, end in CDS_list_init]
        first_CDS_position = CDS_list[0][0]
        gene_attribute = (CDS_list, CDS_score, CDS_score_list, first_CDS_position)
        filtered_gene_list.append(gene_attribute)
    return filtered_gene_list


def process_gene_segment(region, model_prediction_path, min_cds_length, min_cds_score, min_intron_length):
    location_start, location_end, seq_id, strand, prediction_slice, sequence_slice = region
    if location_start is None:
        gene_list = []
    else:
        with h5py.File(model_prediction_path, 'r') as f:
            if strand == 1:
                prediction_slice = f[seq_id]['predictions_forward'][location_start:location_end]
            else:
                prediction_slice = f[seq_id]['predictions_reverse'][location_start:location_end]

        gene_list = decode_gene_structure(
            location_start,
            prediction_slice,
            sequence_slice,
            min_cds_length,
            min_cds_score,
            min_intron_length
        )
    return gene_list, seq_id, strand


def process_gene_segment_batch(regions, model_prediction_path, min_cds_length, min_cds_score, min_intron_length):
    results = []
    for region in regions:
        location_start, location_end, seq_id, strand, prediction_slice, sequence_slice = region
        if location_start is None:
            gene_list = []
        else:
            with h5py.File(model_prediction_path, 'r') as f:
                if strand == 1:
                    prediction_slice = f[seq_id]['predictions_forward'][location_start:location_end]
                else:
                    prediction_slice = f[seq_id]['predictions_reverse'][location_start:location_end]

            gene_list = decode_gene_structure(
                location_start,
                prediction_slice,
                sequence_slice,
                min_cds_length,
                min_cds_score,
                min_intron_length
            )
        results.append((gene_list, seq_id, strand))
    return results


def write_result(file, num, seq_id, result, length, strand):
    CDS_list, CDS_score, CDS_score_list, _ = result
    file.write(f'# Start gene g{num + 1}\n')
    if strand == 1:
        gene_start, gene_end = CDS_list[0][0], CDS_list[-1][1]
        gene_start, gene_end = gene_start + 1, gene_end + 1  # 0-based to 1-based
        file.write(f'{seq_id}\tANNEVO\tgene\t{gene_start}\t{gene_end}\t{CDS_score:.2f}\t+\t.\tID={seq_id}-g{num + 1}\n')
        file.write(f'{seq_id}\tANNEVO\tmRNA\t{gene_start}\t{gene_end}\t.\t+\t.\tID={seq_id}-g{num + 1}.t1;Parent={seq_id}-g{num + 1}\n')
        for i, exon in enumerate(CDS_list):
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
        gene_start = length - (CDS_list[-1][1] + 1) + 1
        gene_end = length - (CDS_list[0][0] + 1) + 1
        file.write(f'{seq_id}\tANNEVO\tgene\t{gene_start}\t{gene_end}\t{CDS_score:.2f}\t-\t.\tID={seq_id}-g{num + 1}\n')
        file.write(f'{seq_id}\tANNEVO\tmRNA\t{gene_start}\t{gene_end}\t.\t-\t.\tID={seq_id}-g{num + 1}.t1;Parent={seq_id}-g{num + 1}\n')
        CDS_num = 0
        for i, exon in enumerate(CDS_list):
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


def batched(iterable, batch_size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def decode_and_write(potential_gene_list, chromosome_length, model_prediction_path, seq_num, cpu_num,
                     min_cds_length, min_cds_score, min_intron_length, output, show_log=False):
    results = []
    batch_size = 16
    batch_iter = batched(potential_gene_list, batch_size)

    with ProcessPoolExecutor(max_workers=cpu_num) as executor:
        future_to_batch = {
            executor.submit(
                process_gene_segment_batch,
                batch,
                model_prediction_path,
                min_cds_length,
                min_cds_score,
                min_intron_length
            ): batch for batch in batch_iter
        }
        future_iter = as_completed(future_to_batch)
        if show_log:
            future_iter = tqdm(future_iter, total=len(future_to_batch), desc='Decoding segment batches')
        for future in future_iter:
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                print(f"Process failed: {str(e)}")
                continue

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


def get_gene_region(chromosome, predictions_forward, predictions_reverse, sequence_forward, sequence_reverse,
                    average_threshold, max_threshold):
    potential_gene_list = []
    length = len(sequence_forward)
    '''
    position index conversion
    The position tuple in the forward array is (a, b) 
    The position tuple in the forward chains of gff is (a + 1, b)
    The position tuple in the reverse chains of gff is (length - b + 1, length - (a + 1) + 1) = (length - b + 1, length - a)
    The position tuple in the reverse array is (length - b, length - a) 
    '''

    potential_gene_chromosome_forward = detect_gene_location(
        predictions_forward,
        length,
        average_threshold,
        max_threshold
    )
    potential_gene_chromosome_forward = expand_and_merge_regions(
        potential_gene_chromosome_forward, length, buffer_size=100
    )
    if not potential_gene_chromosome_forward:
        potential_gene_list.append(
            (None, None, chromosome, 1, None, None)
        )
    else:
        for location_start, location_end in potential_gene_chromosome_forward:
            potential_gene_list.append(
                (location_start, location_end, chromosome, 1,
                 None,
                 sequence_forward[location_start:location_end])
            )

    potential_gene_chromosome_reverse = detect_gene_location(
        predictions_reverse,
        length,
        average_threshold,
        max_threshold
    )
    potential_gene_chromosome_reverse = expand_and_merge_regions(
        potential_gene_chromosome_reverse, length, buffer_size=100
    )
    if not potential_gene_chromosome_reverse:
        potential_gene_list.append(
            (None, None, chromosome, -1, None, None)
        )
    else:
        for location_start, location_end in potential_gene_chromosome_reverse:
            potential_gene_list.append(
                (location_start, location_end, chromosome, -1,
                 None,
                 sequence_reverse[location_start:location_end])
            )

    return length, potential_gene_list


def load_and_detect_gene_region(chr_name, sequence_forward, model_prediction_path, average_threshold, max_threshold):
    sequence_forward = re.sub(r'[^ATCGatcg]', 'N', sequence_forward)
    sequence_reverse = rev_complement(sequence_forward)

    start_time = time.time()
    with h5py.File(model_prediction_path, 'r') as h5file:
        chr_group = h5file[chr_name]
        predictions_forward = np.array(chr_group['predictions_forward'])
        predictions_reverse = np.array(chr_group['predictions_reverse'])
    end_time = time.time()

    chromosome_length_single, potential_gene_list_single = get_gene_region(
        chr_name,
        predictions_forward,
        predictions_reverse,
        sequence_forward,
        sequence_reverse,
        average_threshold,
        max_threshold
    )
    return chr_name, chromosome_length_single, potential_gene_list_single, end_time - start_time


def gene_structure_decoding(genome, model_prediction_path, output, cpu_num, average_threshold, max_threshold,
                            min_cds_length, min_cds_score, min_intron_length, show_log=False, region_threads=None):
    file_loading_time = 0
    chromosome_length = {}
    potential_gene_list = []

    with h5py.File(f'{model_prediction_path}', 'r') as h5file:
        chromosome_names = list(h5file.keys())

    genome_seqIO = SeqIO.index(genome, "fasta")
    with ProcessPoolExecutor(max_workers=region_threads) as executor:
        region_results = {}
        futures = [
            executor.submit(
                load_and_detect_gene_region,
                chr_name,
                str(genome_seqIO[chr_name].seq).upper(),
                model_prediction_path,
                average_threshold,
                max_threshold
            )
            for chr_name in chromosome_names
        ]
        future_iter = as_completed(futures)
        if show_log:
            future_iter = tqdm(future_iter, total=len(futures), desc='Loading data and detecting potential genes')
        for future in future_iter:
            chr_name, chromosome_length_single, potential_gene_list_single, loading_time = future.result()
            file_loading_time += loading_time
            region_results[chr_name] = (chromosome_length_single, potential_gene_list_single)

    for chr_name in chromosome_names:
        chromosome_length_single, potential_gene_list_single = region_results[chr_name]
        chromosome_length[chr_name] = chromosome_length_single
        potential_gene_list.extend(potential_gene_list_single)

    seq_num = 1
    with open(output, 'w') as file:
        file.write('# This output was generated with ANNEVO (v2.3.0).\n')
        file.write('# ANNEVO is an ab initio gene annotation tool written by YeLab.\n')
        file.write('# Citation: Zhang, P., Xu, T., Wang, S. et al. Highly accurate ab initio gene annotation with ANNEVO. Nat Methods (2026). https://doi.org/10.1038/s41592-026-03036-7\n')

    if potential_gene_list:
        decode_and_write(
            potential_gene_list,
            chromosome_length,
            model_prediction_path,
            seq_num,
            cpu_num,
            min_cds_length,
            min_cds_score,
            min_intron_length,
            output,
            show_log=show_log
        )
