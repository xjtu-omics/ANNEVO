import os
import numpy as np
from Bio import SeqIO
from BCBio import GFF
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed


def error_checking_forward(max_mRNA, weights_forward, mask_length, sequences, forward, average_length_5p_UTR, average_length_3p_UTR):
    """
    Because annotations may contain errors or incomplete information, we further check the GFF file and may reset the weights to 0 for incorrect positions.
    Based on the types of errors that can occur, there are roughly five categories (see GitHub for details):
    1. Missing UTR
    2. Empty gene (Already processed above)
    3. Missing codon or wrong codon
    4. Incorrect intron length
    5. CDS phase mismatch
    6. CDS or exon is located outside the mRNA
    7. CDS or exon overlapping
    8. Except for the first and last CDS, the positions of other CDS and exon do not match
    Reference: Part of the content refers to https://github.com/weberlab-hhu/GeenuFF/tree/main
    """
    CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
    exon_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'exon']
    CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start))
    exon_features = sorted(exon_features, key=lambda x: int(x.location.start))
    first_CDS = CDS_features[0]
    last_CDS = CDS_features[-1]
    # Missing 5'UTR
    if int(first_CDS.location.start) <= int(max_mRNA.location.start):
        weights_forward[int(first_CDS.location.start) - 2 * average_length_5p_UTR:int(first_CDS.location.start)] = 0
    # Missing 3'UTR
    if int(last_CDS.location.end) >= int(max_mRNA.location.end):
        weights_forward[int(last_CDS.location.end):int(last_CDS.location.end) + 2 * average_length_3p_UTR] = 0

    # calculate whether intron length < 30 bp
    if len(exon_features) > 1:
        for i in range(len(exon_features) - 1):
            if int(exon_features[i + 1].location.start) - int(exon_features[i].location.end) <= 30:
                weights_forward[int(exon_features[i].location.end):int(exon_features[i + 1].location.start)] = 0

    CDS_seq = ''
    for CDS_feature in CDS_features:
        CDS_seq += sequences[int(CDS_feature.location.start):int(CDS_feature.location.end)]
    # Missing or incorrect start codon
    if CDS_seq[0:3] != 'ATG':
        weights_forward[int(max_mRNA.location.start):int(first_CDS.location.start)] = 0
    # Missing or incorrect stop codon
    if CDS_seq[-3:] not in ['TAA', 'TAG', 'TGA']:
        weights_forward[int(last_CDS.location.end):int(max_mRNA.location.end)] = 0

    # CDS phase mismatch
    if len(CDS_seq) % 3 != 0:
        weights_forward[int(first_CDS.location.start) - mask_length:int(first_CDS.location.start)] = 0
        weights_forward[int(last_CDS.location.end):int(last_CDS.location.end) + mask_length] = 0

    # CDS or exon is located outside the mRNA
    # The processing method and the positions that need to be processed are the same as 'Missing UTR'

    # CDS or exon overlapping
    if len(CDS_features) > 1:
        for i in range(len(CDS_features) - 1):
            CDS_feature = CDS_features[i]
            next_CDS_feature = CDS_features[i + 1]
            if int(CDS_feature.location.end) > int(next_CDS_feature.location.start):
                weights_forward[int(CDS_feature.location.start):int(next_CDS_feature.location.end)] = 0
    if len(exon_features) > 1:
        for i in range(len(exon_features) - 1):
            exon_feature = exon_features[i]
            next_exon_feature = exon_features[i + 1]
            if int(exon_feature.location.end) > int(next_exon_feature.location.start):
                weights_forward[int(exon_feature.location.start):int(next_exon_feature.location.end)] = 0

    # Except for the first and last CDS, the positions of other CDS and exon do not match
    # It is impossible for UTR to appear after CDS starts until it ends.
    weights_forward[int(first_CDS.location.start):int(last_CDS.location.end)][forward[int(first_CDS.location.start):int(last_CDS.location.end)] == 1] = 0

    return weights_forward


def error_checking_reverse(max_mRNA, weights_reverse, mask_length, sequences, reverse, average_length_5p_UTR, average_length_3p_UTR):
    """
    Because annotations may contain errors or incomplete information, we further check the GFF file and may reset the weights to 0 for incorrect positions.
    Based on the types of errors that can occur, there are roughly five categories (see GitHub for details):
    1. Missing UTR
    2. Empty gene (Already solved above)
    3. Missing codon or wrong codon
    4. Incorrect intron length
    5. CDS phase mismatch
    6. CDS or exon is located outside the mRNA
    7. CDS or exon overlapping
    8. Except for the first and last CDS, the positions of other CDS and exon do not match
    Reference: Part of the content refers to https://github.com/weberlab-hhu/GeenuFF/tree/main
    """
    CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
    exon_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'exon']
    CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start), reverse=True)
    exon_features = sorted(exon_features, key=lambda x: int(x.location.start), reverse=True)
    first_CDS = CDS_features[0]
    last_CDS = CDS_features[-1]
    # Missing 5'UTR
    if int(first_CDS.location.end) >= int(max_mRNA.location.end):
        weights_reverse[int(first_CDS.location.end):int(first_CDS.location.end) + 2 * average_length_5p_UTR] = 0
    # Missing 3'UTR
    if int(last_CDS.location.start) <= int(max_mRNA.location.start):
        weights_reverse[int(last_CDS.location.start) - 2 * average_length_3p_UTR:int(last_CDS.location.start)] = 0

    # calculate whether intron length < 30 bp
    if len(exon_features) > 1:
        for i in range(len(exon_features) - 1):
            if int(exon_features[i].location.start) - int(exon_features[i + 1].location.end) <= 30:
                weights_reverse[int(exon_features[i + 1].location.end):int(exon_features[i].location.start)] = 0

    CDS_seq = ''
    for CDS_feature in CDS_features:
        CDS_seq = sequences[int(CDS_feature.location.start):int(CDS_feature.location.end)] + CDS_seq
    # Missing or incorrect start codon
    if CDS_seq[-3:] != 'CAT':
        weights_reverse[int(first_CDS.location.end):int(max_mRNA.location.end)] = 0
    # Missing or incorrect stop codon
    if CDS_seq[0:3] not in ['TTA', 'CTA', 'TCA']:
        weights_reverse[int(max_mRNA.location.start):int(last_CDS.location.start)] = 0

    # CDS phase mismatch
    if len(CDS_seq) % 3 != 0:
        weights_reverse[int(first_CDS.location.end):int(first_CDS.location.end) + mask_length] = 0
        weights_reverse[int(last_CDS.location.start) - mask_length:int(last_CDS.location.start)] = 0

    # CDS or exon is located outside the mRNA
    # The processing logic and the positions that need to be processed are the same as Missing UTR

    # CDS or exon overlapping
    if len(CDS_features) > 1:
        for i in range(len(CDS_features) - 1):
            CDS_feature = CDS_features[i]
            next_CDS_feature = CDS_features[i + 1]
            if int(next_CDS_feature.location.end) > int(CDS_feature.location.start):
                weights_reverse[int(next_CDS_feature.location.start):int(CDS_feature.location.end)] = 0
    if len(exon_features) > 1:
        for i in range(len(exon_features) - 1):
            exon_feature = exon_features[i]
            next_exon_feature = exon_features[i + 1]
            if int(next_exon_feature.location.end) > int(exon_feature.location.start):
                weights_reverse[int(next_exon_feature.location.start):int(exon_feature.location.end)] = 0

    # There is an error if the positions of other CDSs and exons do not match (except the first and last CDS)
    # It is impossible for UTR to appear after CDS starts until it ends.
    weights_reverse[int(last_CDS.location.start):int(first_CDS.location.end)][reverse[int(last_CDS.location.start):int(first_CDS.location.end)] == 1] = 0

    return weights_reverse


def transition_weights_forward(feature, weights_forward, transition_weights):
    """
    In the organism's genome, positions where the category changes usually have more important biological significance, such as start codon, stop codon, splice donors and acceptors.
    Therefore, we give larger weight to these positions to clarify their significance.
    1. intergenic -> UTR (or intergenic -> nc_exon)
    2. UTR -> intergenic (or nc_exon -> intergenic)
    3. UTR -> CDS
    4. CDS -> UTR
    5. CDS -> intron (or UTR-intron, nc_exon -> intron)
    6. intron -> CDS (or intron-UTR, intron -> nc_exon)
    Reference: Part of the content refers to Helixer–de novo Prediction of Primary Eukaryotic Gene Models Combining Deep Learning and a Hidden Markov Model[J].
    bioRxiv, 2023: 2023.02. 06.527280.
    """
    CDS_features = [sub_feature for sub_feature in feature.sub_features if sub_feature.type == 'CDS']
    exon_features = [sub_feature for sub_feature in feature.sub_features if sub_feature.type == 'exon']
    exon_features = sorted(exon_features, key=lambda x: int(x.location.start))
    if CDS_features:
        CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start))
        first_CDS = CDS_features[0]
        last_CDS = CDS_features[-1]
        weights_forward[first_CDS.location.start - 1:first_CDS.location.start + 3] = transition_weights[2]  # UTR-CDS, positions include start codon and the last 5'UTR site
        weights_forward[last_CDS.location.end - 3:last_CDS.location.end + 1] = transition_weights[3]  # CDS-UTR, positions include stop codon and the fist 3'UTR site
    weights_forward[feature.location.start - 1:feature.location.start + 1] = transition_weights[0]  # intergenic -> UTR (or intergenic -> nc_exon)
    weights_forward[feature.location.end - 1:feature.location.end + 1] = transition_weights[1]  # UTR -> intergenic (or nc_exon -> intergenic)
    # There are no intron when there is only one exon
    if len(exon_features) > 1:
        for i, exon_feature in enumerate(exon_features):
            if i == 0:
                # CDS-intron (or UTR-intron), positions include splice donor site and the last CDS (UTR) site
                weights_forward[exon_feature.location.end - 1:exon_feature.location.end + 2] = transition_weights[4]
            elif i == len(exon_features) - 1:
                # intron-CDS (or intron-UTR), positions include splice acceptor site and the first CDS (UTR) site
                weights_forward[exon_feature.location.start - 2:exon_feature.location.start + 1] = transition_weights[5]
            else:
                # CDS-intron (or UTR-intron), positions include splice donor site and the last CDS (UTR) site
                weights_forward[exon_feature.location.end - 1:exon_feature.location.end + 2] = transition_weights[4]
                # intron-CDS (or intron-UTR), positions include splice acceptor site and the first CDS (UTR) site
                weights_forward[exon_feature.location.start - 2:exon_feature.location.start + 1] = transition_weights[5]

    return weights_forward


def transition_weights_reverse(max_mRNA, weights_reverse, transition_weights):
    CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
    exon_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'exon']
    exon_features = sorted(exon_features, key=lambda x: int(x.location.start), reverse=True)
    if CDS_features:
        CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start), reverse=True)
        first_CDS = CDS_features[0]
        last_CDS = CDS_features[-1]
        weights_reverse[first_CDS.location.end - 3:first_CDS.location.end + 1] = transition_weights[2]  # UTR-CDS, positions include start codon and the last 5'UTR site
        weights_reverse[last_CDS.location.start - 1:last_CDS.location.start + 3] = transition_weights[3]  # CDS-UTR, positions include stop codon and the fist 3'UTR site

    weights_reverse[max_mRNA.location.end - 1:max_mRNA.location.end + 1] = transition_weights[0]  # intergenic-UTR
    weights_reverse[max_mRNA.location.start - 1:max_mRNA.location.start + 1] = transition_weights[1]  # UTR-intergenic

    # There are no intron when there is only one exon
    if len(exon_features) > 1:
        for i, exon_feature in enumerate(exon_features):
            if i == 0:
                # CDS-intron (or UTR-intron), positions include splice donor site and the last CDS (UTR) site
                weights_reverse[exon_feature.location.start - 2:exon_feature.location.start + 1] = transition_weights[4]
            elif i == len(exon_features) - 1:
                # intron-CDS (or intron-UTR), positions include splice acceptor site and the first CDS (UTR) site
                weights_reverse[exon_feature.location.end - 1:exon_feature.location.end + 2] = transition_weights[5]
            else:
                # CDS-intron (or UTR-intron), positions include splice donor site and the last CDS (UTR) site
                weights_reverse[exon_feature.location.start - 2:exon_feature.location.start + 1] = transition_weights[4]
                # intron-CDS (or intron-UTR), positions include splice acceptor site and the first CDS (UTR) site
                weights_reverse[exon_feature.location.end - 1:exon_feature.location.end + 2] = transition_weights[5]

    return weights_reverse


def calculate_average_utr_length(seq_information):
    _, sequence, features = seq_information
    length_5p_UTR = 0
    valid_gene_number = 0
    length_3p_UTR = 0
    base_annotation_forward_rec = np.zeros(len(sequence), dtype=np.uint8)
    base_annotation_reverse_rec = np.zeros(len(sequence), dtype=np.uint8)
    for feature in features:
        if feature.type == 'gene' and feature.sub_features and feature.strand != '.':
            exist_mRNA = False
            max_mRNA_len = 0
            max_mRNA = None
            for sub_feature in feature.sub_features:
                if sub_feature.type == "mRNA":
                    exist_mRNA = True
                    mRNA_len = len(sub_feature)
                    if mRNA_len > max_mRNA_len:
                        max_mRNA_len = mRNA_len
                        max_mRNA = sub_feature
            if exist_mRNA:
                CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
                CDS_features.sort(key=lambda x: x.location.start)
                if not CDS_features:
                    continue
                if feature.location.strand == 1:
                    base_annotation_forward_rec[max_mRNA.location.start:max_mRNA.location.end] = 3
                    for sub_feature in max_mRNA.sub_features:
                        if sub_feature.type == 'CDS':
                            base_annotation_forward_rec[sub_feature.location.start:sub_feature.location.end] = 2
                    for sub_feature in max_mRNA.sub_features:
                        if sub_feature.type == 'exon':
                            # The exon is a UTR if not a CDS
                            UTR_region = base_annotation_forward_rec[sub_feature.location.start:sub_feature.location.end] != 2
                            base_annotation_forward_rec[sub_feature.location.start:sub_feature.location.end][UTR_region] = 1
                    length_5p_UTR_rec = np.sum(base_annotation_forward_rec[max_mRNA.location.start:CDS_features[0].location.start] == 1)
                    length_3p_UTR_rec = np.sum(base_annotation_forward_rec[CDS_features[-1].location.end:max_mRNA.location.end] == 1)
                    if length_5p_UTR_rec > 0 and length_3p_UTR_rec > 0:
                        length_5p_UTR += length_5p_UTR_rec
                        length_3p_UTR += length_3p_UTR_rec
                        valid_gene_number += 1
                else:
                    base_annotation_reverse_rec[max_mRNA.location.start:max_mRNA.location.end] = 3
                    for sub_feature in max_mRNA.sub_features:
                        if sub_feature.type == 'CDS':
                            base_annotation_reverse_rec[sub_feature.location.start:sub_feature.location.end] = 2
                    for sub_feature in max_mRNA.sub_features:
                        if sub_feature.type == 'exon':
                            UTR_region = base_annotation_reverse_rec[sub_feature.location.start:sub_feature.location.end] != 2
                            base_annotation_reverse_rec[sub_feature.location.start:sub_feature.location.end][UTR_region] = 1
                    length_5p_UTR_rec = np.sum(base_annotation_forward_rec[CDS_features[-1].location.end:max_mRNA.location.end] == 1)
                    length_3p_UTR_rec = np.sum(base_annotation_forward_rec[max_mRNA.location.start:CDS_features[0].location.start] == 1)
                    if length_5p_UTR_rec > 0 and length_3p_UTR_rec > 0:
                        length_5p_UTR += length_5p_UTR_rec
                        length_3p_UTR += length_3p_UTR_rec
                        valid_gene_number += 1

    return length_5p_UTR, length_3p_UTR, valid_gene_number


def fill_phase(arr, start, end, phase, strand):
    # 确定填充序列
    pattern = [0, 2, 1]

    # 根据 phase 调整起始位置
    phase_index = pattern.index(phase)
    adjusted_pattern = pattern[phase_index:] + pattern[:phase_index]
    # 填充数组
    length = end - start
    if strand == 1:
        for i in range(length):
            arr[start + i] = adjusted_pattern[i % 3]
    else:
        for i in range(length):
            arr[end - i - 1] = adjusted_pattern[i % 3]
    return arr


def parse_files(seq_information, average_5p_UTR, average_3p_UTR, transition_weights, mask_length, window_size, flank_length, discard_all_intergenic):
    """
    The label of each position represents the class of every position. The definition of label are as follows:
    0: intergenic
    1: UTR
    2: CDS
    3: intron
    The basic logic of assigning labels:
    First initialize all to intergenic, then initialize all internal regions of the gene to intron,
    then label the CDS region as CDS, and finally label the exon that is not CDS as UTR.
    """
    seq_id, sequence, features = seq_information
    base_annotation_forward_rec = np.zeros(len(sequence), dtype=np.uint8)
    base_annotation_reverse_rec = np.zeros(len(sequence), dtype=np.uint8)
    phase_annotation_forward_rec = np.zeros(len(sequence), dtype=np.uint8) + 3  # Annotation of a single chromosome, 0-3 correspond to 'Phase0', 'Phase2', 'Phase1' and 'None', respectively.
    phase_annotation_reverse_rec = np.zeros(len(sequence), dtype=np.uint8) + 3
    base_weights_forward_rec = np.ones(len(sequence), dtype=np.uint8)  # Weights of a single chromosome, 0 represents a potential error, 1 represents the normal base,
    base_weights_reverse_rec = np.ones(len(sequence), dtype=np.uint8)  # values greater than 1 represent the weight of the transition position
    record_location_forward = 0
    record_location_reverse = 0
    for feature in features:
        if feature.type in ['pseudogene', 'cDNA_match', 'ncRNA_gene']:
            if feature.location.strand == 1:
                base_weights_forward_rec[feature.location.start:feature.location.end] = 0
            elif feature.location.strand == -1:
                base_weights_reverse_rec[feature.location.start:feature.location.end] = 0

        elif feature.type == 'gene':
            if not feature.sub_features:
                if feature.location.strand == 1:
                    base_weights_forward_rec[feature.location.start:feature.location.end] = 0
                elif feature.location.strand == -1:
                    base_weights_reverse_rec[feature.location.start:feature.location.end] = 0
            else:
                exist_mRNA = False
                max_mRNA_len = 0
                max_mRNA = None
                for sub_feature in feature.sub_features:
                    if sub_feature.type in ['mRNA', 'V_gene_segment', 'C_gene_segment', 'J_gene_segment', 'D_gene_segment']:
                        exist_mRNA = True
                        mRNA_len = len(sub_feature)
                        if mRNA_len > max_mRNA_len:
                            max_mRNA_len = mRNA_len
                            max_mRNA = sub_feature
                if exist_mRNA:
                    CDS_features = [sub_feature for sub_feature in max_mRNA.sub_features if sub_feature.type == 'CDS']
                    if feature.location.strand == 1:
                        if max_mRNA.location.start < record_location_forward:
                            base_weights_forward_rec[feature.location.start:feature.location.end] = 0
                            continue
                        if not CDS_features:
                            # In rare case, there is no CDS region within the mRNA in the GFF file.
                            base_weights_forward_rec[feature.location.start:feature.location.end] = 0
                            continue
                        record_location_forward = max_mRNA.location.end

                        base_annotation_forward_rec[max_mRNA.location.start:max_mRNA.location.end] = 3
                        for sub_feature in max_mRNA.sub_features:
                            if sub_feature.type == 'CDS':
                                phase = int(sub_feature.qualifiers.get('phase')[0])
                                phase_annotation_forward_rec = fill_phase(phase_annotation_forward_rec, sub_feature.location.start, sub_feature.location.end, phase, strand=1)
                                base_annotation_forward_rec[sub_feature.location.start:sub_feature.location.end] = 2
                        for sub_feature in max_mRNA.sub_features:
                            if sub_feature.type == 'exon':
                                # The exon is a UTR if not a CDS
                                UTR_region = base_annotation_forward_rec[sub_feature.location.start:sub_feature.location.end] != 2
                                base_annotation_forward_rec[sub_feature.location.start:sub_feature.location.end][UTR_region] = 1
                        base_weights_forward_rec = transition_weights_forward(max_mRNA, base_weights_forward_rec, transition_weights)
                        base_weights_forward_rec = error_checking_forward(max_mRNA, base_weights_forward_rec, mask_length, sequence, base_annotation_forward_rec, average_5p_UTR, average_3p_UTR)
                    elif feature.location.strand == -1:
                        if max_mRNA.location.start < record_location_reverse:
                            base_weights_reverse_rec[feature.location.start:feature.location.end] = 0
                            continue
                        if not CDS_features:
                            base_weights_reverse_rec[feature.location.start:feature.location.end] = 0
                            continue
                        record_location_reverse = max_mRNA.location.end
                        base_annotation_reverse_rec[max_mRNA.location.start:max_mRNA.location.end] = 3
                        for sub_feature in max_mRNA.sub_features:
                            if sub_feature.type == 'CDS':
                                phase = int(sub_feature.qualifiers.get('phase')[0])
                                phase_annotation_reverse_rec = fill_phase(phase_annotation_reverse_rec, sub_feature.location.start, sub_feature.location.end, phase, strand=-1)
                                base_annotation_reverse_rec[sub_feature.location.start:sub_feature.location.end] = 2
                        for sub_feature in max_mRNA.sub_features:
                            if sub_feature.type == 'exon':
                                UTR_region = base_annotation_reverse_rec[sub_feature.location.start:sub_feature.location.end] != 2
                                base_annotation_reverse_rec[sub_feature.location.start:sub_feature.location.end][UTR_region] = 1
                        base_weights_reverse_rec = transition_weights_reverse(max_mRNA, base_weights_reverse_rec, transition_weights)
                        base_weights_reverse_rec = error_checking_reverse(max_mRNA, base_weights_reverse_rec, mask_length, sequence, base_annotation_reverse_rec, average_5p_UTR, average_3p_UTR)
                else:
                    if feature.location.strand == 1:
                        base_weights_forward_rec[feature.location.start:feature.location.end] = 0
                    elif feature.location.strand == -1:
                        base_weights_reverse_rec[feature.location.start:feature.location.end] = 0

    windows = split_sequence(sequence, base_annotation_forward_rec, base_annotation_reverse_rec, phase_annotation_forward_rec, phase_annotation_reverse_rec,
                             base_weights_forward_rec, base_weights_reverse_rec, window_size, flank_length, discard_all_intergenic)

    return seq_id, windows


def reverse_complement(dna_sequence):
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N', 'X': 'X',
                  'Y': 'Y', 'R': 'R', 'M': 'M', 'W': 'W', 'K': 'K', 'B': 'B', 'S': 'S', 'D': 'D', 'H': 'H', 'V': 'V'}
    return ''.join(complement[nucleotide] for nucleotide in reversed(dna_sequence))


def calculate_state_transition(labels, weights):
    """
    The label of each position determines the state transition from the current position to the next position. The definition of label are as follows:
    0 means the next base state remains unchanged, 1 means a change.
    """
    transition_annotation = np.zeros_like(labels, dtype=np.uint8)
    transition_mask = np.ones_like(weights, dtype=np.uint8)
    state_num = 4
    transition_matrix = np.array([
        0, 1, 0, 0,  # intergenic to other state
        1, 0, 1, 1,  # UTR to other state
        0, 1, 0, 1,  # CDS to other state
        0, 1, 1, 0  # intron to other state
    ]).reshape((state_num, state_num))

    # 将标签映射到转换矩阵中
    for i in range(state_num):
        current_position = (labels[:-1] == i)

        for j in range(state_num):
            next_position = (labels[1:] == j)
            transition_annotation[:-1][current_position & next_position] = transition_matrix[i, j]

    zero_weights_mask = weights == 0
    shifted_zero_weights_mask = np.roll(zero_weights_mask, shift=-1)
    combined_mask = zero_weights_mask | shifted_zero_weights_mask
    transition_mask[combined_mask] = 0

    transition_mask[-1] = 0

    return transition_annotation, transition_mask


def split_sequence(sequence_forward, annotation_forward, annotation_reverse, phase_forward, phase_reverse,
                   weight_forward, weight_reverse, window_size, flank_length, discard_all_intergenic):
    length = len(sequence_forward)
    windows = []
    for start in range(0, length, window_size):
        end = start + window_size
        if start - flank_length < 0:
            if end + flank_length <= length:
                pad_before = 'X' * (flank_length - start)
                window_seq_forward = pad_before + sequence_forward[0:end + flank_length]
            else:
                pad_before = 'X' * (flank_length - start)
                pad_after = 'X' * (end + flank_length - length)
                window_seq_forward = pad_before + sequence_forward[0:length] + pad_after
        elif end + flank_length > length:
            pad_after = 'X' * (end + flank_length - length)
            window_seq_forward = sequence_forward[start - flank_length:length] + pad_after
        else:
            window_seq_forward = sequence_forward[start - flank_length:end + flank_length]
        window_seq_reverse = reverse_complement(window_seq_forward)

        if end > length:
            window_ann_forward = annotation_forward[start:length]
            window_ann_forward = np.pad(window_ann_forward, (0, window_size - len(window_ann_forward)), 'constant')
            window_ann_reverse = annotation_reverse[start:length]
            window_ann_reverse = np.pad(window_ann_reverse, (0, window_size - len(window_ann_reverse)), 'constant')
            window_ann_reverse = window_ann_reverse[::-1]

            window_phase_forward = phase_forward[start:length]
            window_phase_forward = np.pad(window_phase_forward, (0, window_size - len(window_phase_forward)), 'constant', constant_values=(3, 3))
            window_phase_reverse = phase_reverse[start:length]
            window_phase_reverse = np.pad(window_phase_reverse, (0, window_size - len(window_phase_reverse)), 'constant', constant_values=(3, 3))
            window_phase_reverse = window_phase_reverse[::-1]

            window_weights_forward = weight_forward[start:length]
            window_weights_forward = np.pad(window_weights_forward, (0, window_size - len(window_weights_forward)), 'constant')
            window_weights_reverse = weight_reverse[start:length]
            window_weights_reverse = np.pad(window_weights_reverse, (0, window_size - len(window_weights_reverse)), 'constant')
            window_weights_reverse = window_weights_reverse[::-1]
        else:
            window_ann_forward = annotation_forward[start:end]
            window_ann_reverse = annotation_reverse[start:end][::-1]

            window_phase_forward = phase_forward[start:end]
            window_phase_reverse = phase_reverse[start:end][::-1]

            window_weights_forward = weight_forward[start:end]
            window_weights_reverse = weight_reverse[start:end][::-1]

        transition_annotation_forward, transition_mask_forward = calculate_state_transition(window_ann_forward, window_weights_forward)
        transition_annotation_reverse, transition_mask_reverse = calculate_state_transition(window_ann_reverse, window_weights_reverse)

        if discard_all_intergenic:
            if not (np.all(annotation_forward[start:end] == 0) or np.all(weight_forward[start:end] == 0)):
                windows.append((window_seq_forward, window_ann_forward, window_weights_forward, transition_annotation_forward, transition_mask_forward, window_phase_forward))
            if not (np.all(annotation_reverse[start:end] == 0) or np.all(weight_reverse[start:end] == 0)):
                windows.append((window_seq_reverse, window_ann_reverse, window_weights_reverse, transition_annotation_reverse, transition_mask_reverse, window_phase_reverse))
        else:
            if not np.all(weight_forward[start:end] == 0):
                windows.append((window_seq_forward, window_ann_forward, window_weights_forward, transition_annotation_forward, transition_mask_forward, window_phase_forward))
            if not np.all(weight_reverse[start:end] == 0):
                windows.append((window_seq_reverse, window_ann_reverse, window_weights_reverse, transition_annotation_reverse, transition_mask_reverse, window_phase_reverse))
    return windows


def create_dataset(genome, annotation, output_file, cpu_num, window_size, flank_length, transition_weights, mask_length, discard_all_intergenic):
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    seq_information_total = []
    seq_information_total_exceed_max = []
    max_overflow_length = 300000000
    with open(genome) as genome_data:
        genome_seq = SeqIO.to_dict(SeqIO.parse(genome_data, "fasta"))
    with open(annotation) as gff_handle:
        for rec in GFF.parse(gff_handle):
            features = []
            sequence = str(genome_seq[rec.id].seq).upper()
            for feature in rec.features:
                features.append(feature)
            if len(sequence) > max_overflow_length:
                seq_information_total_exceed_max.append((rec.id, sequence, features))
            else:
                seq_information_total.append((rec.id, sequence, features))

    results = [calculate_average_utr_length(seq_information) for seq_information in seq_information_total_exceed_max]
    with ProcessPoolExecutor(max_workers=cpu_num) as executor:
        future_to_segment = {executor.submit(calculate_average_utr_length, seq_information): seq_information for seq_information in seq_information_total}
        for future in as_completed(future_to_segment):
            result = future.result()
            results.append(result)
    length_5p_UTR_sum = 0
    length_3p_UTR_sum = 0
    gene_number_sum = 0
    for length_5p_UTR, length_3p_UTR, gene_number in results:
        length_5p_UTR_sum += length_5p_UTR
        length_3p_UTR_sum += length_3p_UTR
        gene_number_sum += gene_number
    # In some species, all genes in the annotations have no UTR. Use a mask of length 500 instead.
    if gene_number_sum == 0:
        average_5p_UTR = int(mask_length / 2)
        average_3p_UTR = int(mask_length / 2)
    else:
        average_5p_UTR = int(length_5p_UTR_sum / gene_number_sum)
        average_3p_UTR = int(length_3p_UTR_sum / gene_number_sum)
    print(f'The length of average 5p UTR is {average_5p_UTR}')
    print(f'The length of average 3p UTR is {average_3p_UTR}')

    results = [parse_files(seq_information, average_5p_UTR, average_3p_UTR, transition_weights, mask_length, window_size, flank_length, discard_all_intergenic)
               for seq_information in seq_information_total_exceed_max]
    with ProcessPoolExecutor(max_workers=cpu_num) as executor:
        future_to_segment = {executor.submit(parse_files, seq_information, average_5p_UTR, average_3p_UTR, transition_weights, mask_length, window_size, flank_length,
                                             discard_all_intergenic): seq_information for seq_information in seq_information_total}
        for future in as_completed(future_to_segment):
            result = future.result()
            results.append(result)

    split_genome = {}
    for result in results:
        seq_id, windows = result
        split_genome[seq_id] = windows

    with h5py.File(output_file, "w") as f:
        for chromosome, windows in split_genome.items():
            grp = f.create_group(str(chromosome))
            sequences = [window[0] for window in windows]
            annotations = np.array([window[1] for window in windows])
            weights = np.array([window[2] for window in windows])
            transition_annotation = np.array([window[3] for window in windows])
            transition_mask = np.array([window[4] for window in windows])
            phases = np.array([window[5] for window in windows])

            dt = h5py.special_dtype(vlen=str)
            grp.create_dataset("sequences", data=sequences, dtype=dt)

            grp.create_dataset("annotations", data=annotations)
            grp.create_dataset("weights", data=weights)
            grp.create_dataset("transition_annotation", data=transition_annotation)
            grp.create_dataset("transition_mask", data=transition_mask)
            grp.create_dataset("phases", data=phases)
