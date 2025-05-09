import numpy as np
from Bio import SeqIO
import h5py
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
from src.HMM import viterbi_decoding, define_state


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
                if count_above_threshold >= 1:
                    genic_region.append((genic_region_start, end))
                # above_threshold = np.any(potential_genic_range > max_threshold)
                # if above_threshold:
                #     genic_region.append((genic_region_start, end))
                in_genic_region = False
            genic_region_start = start + step
    if in_genic_region:
        genic_region.append((genic_region_start, seq_length))
    return genic_region


def calculate_gene_score(gene_structure, predictions, CDS_list, intron_list, sequence, CDS_phase0, CDS_phase2, CDS_phase1):
    CDS_score = 0
    CDS_num = 0

    CDS_score_list = []

    sequence = sequence.upper()
    for CDS_start, CDS_end in CDS_list:
        CDS_score_single_sum = 0
        CDS_num_single = CDS_end - CDS_start + 1
        coefficient = 0
        for i in range(CDS_start, CDS_end + 1):
            if gene_structure[i] in CDS_phase0:
                if sequence[i].islower():
                    CDS_score += predictions[i, 1] * coefficient
                    CDS_score_single_sum += predictions[i, 1] * coefficient
                else:
                    CDS_score += predictions[i, 1]
                    CDS_score_single_sum += predictions[i, 1]
                CDS_num += 1
            elif gene_structure[i] in CDS_phase2:
                if sequence[i].islower():
                    CDS_score += predictions[i, 3] * coefficient
                    CDS_score_single_sum += predictions[i, 3] * coefficient
                else:
                    CDS_score += predictions[i, 3]
                    CDS_score_single_sum += predictions[i, 3]
                CDS_num += 1
            elif gene_structure[i] in CDS_phase1:
                if sequence[i].islower():
                    CDS_score += predictions[i, 2] * coefficient
                    CDS_score_single_sum += predictions[i, 2] * coefficient
                else:
                    CDS_score += predictions[i, 2]
                    CDS_score_single_sum += predictions[i, 2]
                CDS_num += 1
        CDS_score_single = CDS_score_single_sum / CDS_num_single
        CDS_score_list.append(CDS_score_single)
    CDS_score = CDS_score / CDS_num if CDS_num != 0 else 0
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

    intron_group = ['CDS0', 'CDS0_T', 'CDS1', 'CDS1_TA', 'CDS1_TG', 'CDS2']

    min_intron_length_rare = 1
    states_to_num, num_states, phase_0_columns, phase_1_columns, phase_2_columns, intron_columns = define_state(intron_group, min_intron_length, min_intron_length_rare)
    gene_structure_all_states = viterbi_decoding(predictions, sequence, states_to_num, num_states, phase_0_columns, phase_1_columns, phase_2_columns,
                                                 intron_columns, intron_group, min_intron_length_rare, min_intron_length)
    # print(gene_structure_all_states[0:100])
    CDS_columns = phase_0_columns + phase_2_columns + phase_1_columns
    gene_structure_three_states = [
        0 if x in {states_to_num['intergenic']} else
        1 if x in CDS_columns else
        2 if x in intron_columns else x
        for x in gene_structure_all_states
    ]
    targets = [1, 2]
    gene_list = parse_ranges(gene_structure_three_states, targets)
    filtered_gene_list = []
    for gene in gene_list:
        CDS_list_init = gene['CDS']
        intron_list_init = gene['intron']

        if not CDS_list_init:
            continue
        CDS_count = sum((CDS[1] - CDS[0] + 1) for CDS in CDS_list_init)
        if CDS_count < min_cds_length:
            continue
        CDS_score, CDS_score_list = calculate_gene_score(gene_structure_all_states, predictions, CDS_list_init,
                                                         intron_list_init, sequence, phase_0_columns, phase_2_columns, phase_1_columns)
        if CDS_score < min_cds_score:
            continue

        CDS_list = [(start + location_start, end + location_start) for start, end in CDS_list_init]
        intron_list = [(start + location_start, end + location_start) for start, end in intron_list_init]
        first_CDS_position = CDS_list[0][0]
        gene_length = CDS_list[-1][-1] - first_CDS_position + 1
        if gene_length < 200:
            continue
        gene_attribute = (CDS_list, intron_list, CDS_score, CDS_score_list, first_CDS_position)
        filtered_gene_list.append(gene_attribute)
    return filtered_gene_list


def process_gene_segment(region, min_cds_length, min_cds_score, min_intron_length):
    location_start, location_end, seq_id, strand, prediction_slice, sequence_slice = region
    if location_start is None:
        gene_list = []
    else:
        gene_list = decode_gene_structure(
            location_start,
            prediction_slice,
            sequence_slice,
            min_cds_length,
            min_cds_score,
            min_intron_length
        )
    return gene_list, seq_id, strand


def write_result(file, num, seq_id, result, length, strand):
    CDS_list, intron_list, CDS_score, CDS_score_list, _ = result
    file.write(f'# Start gene g{num + 1}\n')
    file.write(f'# The CDS score is {CDS_score}\n')
    if strand == 1:
        gene_start, gene_end = CDS_list[0][0], CDS_list[-1][1]
        gene_start, gene_end = gene_start + 1, gene_end + 1  # 0-based to 1-based
        file.write(f'{seq_id}\tANNEVO\tgene\t{gene_start}\t{gene_end}\t.\t+\t.\tID={seq_id}-g{num + 1}\n')
        file.write(f'{seq_id}\tANNEVO\ttranscript\t{gene_start}\t{gene_end}\t.\t+\t.\tID={seq_id}-g{num + 1}.t1;Parent={seq_id}-g{num + 1}\n')
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
        file.write(f'{seq_id}\tANNEVO\tgene\t{gene_start}\t{gene_end}\t.\t-\t.\tID={seq_id}-g{num + 1}\n')
        file.write(f'{seq_id}\tANNEVO\ttranscript\t{gene_start}\t{gene_end}\t.\t-\t.\tID={seq_id}-g{num + 1}.t1;Parent={seq_id}-g{num + 1}\n')
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


def gene_structure_decoding(genome, model_prediction_path, output, cpu_num, average_threshold, max_threshold, min_cds_length, min_cds_score, min_intron_length):
    with open(genome) as fna:
        genome_seq = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))
    with open(output, 'w') as file:
        file.write('# This output was generated with ANNEVO (version 2.0).\n')
        file.write('# ANNEVO is a gene prediction tool written by YeLab.\n')
    seq_num = 1
    prediction_files = [f for f in os.listdir(model_prediction_path) if os.path.isfile(os.path.join(model_prediction_path, f))]

    for prediction_file in prediction_files:
        genome_predictions = {}
        with h5py.File(f'{model_prediction_path}/{prediction_file}', 'r') as h5file:
            for chromosome in h5file.keys():
                data = []
                chr_group = h5file[chromosome]
                labels = ['predictions_forward', 'predictions_reverse']
                for label in labels:
                    dataset = np.array(chr_group[label])
                    data.append(dataset)
                genome_predictions[chromosome] = data
        potential_gene_list = []
        chromosome_length = {}

        for chromosome in genome_predictions:
            # if chromosome != 'NC_000002.12':
            #     print(chromosome)
            #     continue
            chromosome_seq_record = genome_seq[chromosome]
            sequence_forward = str(chromosome_seq_record.seq)
            sequence_reverse = reverse_complement(sequence_forward)
            length = len(sequence_forward)
            chromosome_length[chromosome] = length
            predictions_forward, predictions_reverse = genome_predictions[chromosome]

            '''
            position index conversion
            The position tuple in the forward array is (a, b) 
            The position tuple in the forward chains of gff is (a + 1, b)
            The position tuple in the reverse chains of gff is (length - b + 1, length - (a + 1) + 1) = (length - b + 1, length - a)
            The position tuple in the reverse array is (length - b, length - a) 
            '''

            potential_gene_chromosome_forward = detect_gene_location(predictions_forward, length, average_threshold, max_threshold)
            # print(potential_gene_chromosome_forward)
            # potential_gene_chromosome_forward = [(198815, 211346)]
            if not potential_gene_chromosome_forward:
                potential_gene_list.append(
                    (None, None, chromosome, 1, None, None)
                )
            else:
                for location_start, location_end in potential_gene_chromosome_forward:
                    potential_gene_list.append(
                        (location_start, location_end, chromosome, 1,
                         predictions_forward[location_start:location_end],
                         sequence_forward[location_start:location_end])
                    )
            potential_gene_chromosome_reverse = detect_gene_location(predictions_reverse, length, average_threshold, max_threshold)
            # potential_gene_chromosome_reverse = []
            if not potential_gene_chromosome_reverse:
                potential_gene_list.append(
                    (None, None, chromosome, -1, None, None)
                )
            else:
                for location_start, location_end in potential_gene_chromosome_reverse:
                    potential_gene_list.append(
                        (location_start, location_end, chromosome, -1,
                         predictions_reverse[location_start:location_end],
                         sequence_reverse[location_start:location_end])
                    )

        results = []
        with ProcessPoolExecutor(max_workers=cpu_num) as executor:
            future_to_segment = {executor.submit(process_gene_segment, region, min_cds_length, min_cds_score, min_intron_length): region for region in potential_gene_list}
            for future in as_completed(future_to_segment):
                try:
                    result = future.result()
                    results.append(result)
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
