
import numpy as np
import warnings


INTERGENIC = 0
CODING_EXON_0 = 1
CODING_EXON_1 = 2
CODING_EXON_2 = 3
INTRON_0 = 4
INTRON_1 = 5
INTRON_2 = 6
DSS_0 = 7
DSS_1 = 8
DSS_2 = 9
ASS_0 = 10
ASS_1 = 11
ASS_2 = 12
START = 13
END = 14


def define_coding_exon(arr, start, end, phase, strand):
    pattern = [0, 2, 1]

    phase_index = pattern.index(phase)
    adjusted_pattern = pattern[phase_index:] + pattern[:phase_index]
    length = end - start
    if strand == 1:
        for i in range(length):
            arr[start + i] = adjusted_pattern[i % 3] + 1
    else:
        for i in range(length):
            arr[end - i - 1] = adjusted_pattern[i % 3] + 1
    return arr


def get_ordered_cds_features(cds_features, strand):
    return sorted(cds_features, key=lambda x: int(x.location.start), reverse=(strand == -1))


def get_cds_phase_positions(cds_features, strand):
    ordered_cds = get_ordered_cds_features(cds_features, strand)
    positions = []
    for cds in ordered_cds:
        start = int(cds.location.start)
        end = int(cds.location.end)
        if strand == 1:
            positions.extend(range(start, end))
        else:
            positions.extend(range(end - 1, start - 1, -1))
    return positions


def mark_positions(arr, positions, label):
    for pos in positions:
        if 0 <= pos < len(arr):
            arr[pos] = label


def get_phase_from_exon_label(label):
    if label not in (CODING_EXON_0, CODING_EXON_1, CODING_EXON_2):
        print(label)
        raise ValueError(f"Unexpected exon label at splice boundary: {label}")
    return label - 1


def label_intron_and_splice_sites(arr, left_cds, right_cds, strand):
    if strand == 1:
        intron_start = int(left_cds.location.end)
        intron_end = int(right_cds.location.start)
        dss_pos = intron_start - 1
        ass_pos = intron_end
    else:
        intron_start = int(right_cds.location.end)
        intron_end = int(left_cds.location.start)
        dss_pos = intron_end
        ass_pos = intron_start - 1

    if intron_end <= intron_start:
        return

    donor_phase = get_phase_from_exon_label(int(arr[dss_pos]))
    acceptor_phase = get_phase_from_exon_label(int(arr[ass_pos]))
    arr[intron_start:intron_end] = INTRON_0 + donor_phase
    arr[dss_pos] = DSS_0 + donor_phase
    arr[ass_pos] = ASS_0 + acceptor_phase


def define_transcript_label(transcript, ann, record_location):
    if transcript.type == 'mRNA':
        CDS_features = [sub_feature for sub_feature in transcript.sub_features if sub_feature.type == 'CDS']
        CDS_features = sorted(CDS_features, key=lambda x: int(x.location.start))
        if not CDS_features:
            warnings.warn(f'mRNA {transcript.id} has no CDS. Skip this transcript.')
        else:
            if len(CDS_features) >= 3:
                middle_cds = CDS_features[1:-1]
                if any((int(cds.location.end) - int(cds.location.start)) == 1 for cds in middle_cds):
                    warnings.warn(f'mRNA {transcript.id} has an internal CDS of length 1. Skip this transcript.')
                    return ann, record_location

            strand = transcript.location.strand
            CDS_region_end = CDS_features[-1].location.end
            record_location = CDS_region_end
            for sub_feature in CDS_features:
                phase = int(sub_feature.qualifiers.get('phase')[0])
                ann = define_coding_exon(ann, sub_feature.location.start, sub_feature.location.end, phase, strand=strand)

            ordered_cds = get_ordered_cds_features(CDS_features, strand)
            for left_cds, right_cds in zip(ordered_cds, ordered_cds[1:]):
                label_intron_and_splice_sites(ann, left_cds, right_cds, strand)

            cds_positions = get_cds_phase_positions(CDS_features, strand)
            mark_positions(ann, cds_positions[:3], START)
            mark_positions(ann, cds_positions[-3:], END)

    else:
        # Non-coding transcript
        pass

    return ann, record_location


def parse_files(seq_information):
    """
    The label of each position represents the class of every position. The definition of label are as follows:
    0: Intergenic
    1: Coding_exon 0
    2: Coding_exon 1
    3: Coding_exon 2
    4: Intron 0
    5: Intron 1
    6: Intron 2
    7: DSS 0
    8: DSS 1
    9: DSS 2
    10: ASS 0
    11: ASS 1
    12: ASS 2
    13: start
    14: end
    """
    seq_id, sequence, features = seq_information
    seq_length = len(sequence)
    ann_fwd = np.zeros(seq_length, dtype=np.uint8)
    ann_rev = np.zeros(seq_length, dtype=np.uint8)
    loc_fwd = 0
    loc_rev = 0
    valid_transcript_type = ['miRNA', 'primary_transcript', 'lnc_RNA', 'snRNA', 'transcript', 'snoRNA', 'RNA', 'antisense_RNA', 'tRNA',
                             'rRNA', 'RNase_MRP_RNA', 'scRNA', 'RNase_P_RNA', 'mRNA']
    features = sorted(features, key=lambda x: x.location.start)
    for feature in features:
        if feature.location.end > seq_length:
            continue
        if feature.type in ['gene']:
            transcript_list = [f for f in feature.sub_features if f.type in valid_transcript_type]
            if not transcript_list:
                 continue
            transcript_type = [f.type for f in transcript_list]
            if 'mRNA' in transcript_type:
                if transcript_type.count("mRNA") >= 2:
                    warnings.warn(f'{feature.id} has multiple mRNAs. Only consider the first mRNA.')
            else:
                # This version we do not consider non-coding transcripts.
                continue
                if len(transcript_type) >= 2:
                    warnings.warn(f'{feature.id} has multiple non coding transcript. Only consider the first transcript.')

            transcript = transcript_list[0]
            if feature.location.strand == 1:
                if transcript.location.start < loc_fwd:
                    continue
                else:
                    ann_fwd, loc_fwd = define_transcript_label(transcript, ann_fwd, loc_fwd)

            else:
                if transcript.location.start < loc_rev:
                    continue
                else:
                    ann_rev, loc_rev = define_transcript_label(transcript, ann_rev, loc_rev)

    return seq_id, sequence, ann_fwd, ann_rev
