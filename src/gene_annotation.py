from src.utils import model_construction, model_load_weights
from src.predict_nucleotide import reverse_complement, predict_probability
from src.gene_decoding import detect_gene_location, process_gene_segment, write_result
from Bio import SeqIO
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import gc
from collections import defaultdict


def pred_and_decode(genome, lineage, chunk_num, output, threads, num_workers, batch_size, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads,
                    num_blocks, num_branches, average_threshold, max_threshold, min_cds_length, min_cds_score, min_intron_length, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(threads)
    with open(genome) as fna:
        genome_seq = SeqIO.to_dict(SeqIO.parse(fna, "fasta"))

    model = model_construction(device, window_size, flank_length, channels, dim_feedforward, num_encoder_layers, num_heads, num_blocks, num_branches, num_classes, top_k=2)
    model = model_load_weights(lineage, model, device)
    model.eval()

    chromosome_name = []
    chromosome_length = []
    for chromosome in genome_seq:
        chromosome_name.append(chromosome)
        chromosome_seq_record = genome_seq[chromosome]
        sequence = str(chromosome_seq_record.seq).upper()
        length = len(sequence)
        chromosome_length.append(length)
    chunk_num = min(chunk_num, len(chromosome_name))
    print(f'The number of sequence in this species is {len(chromosome_name)}')
    print(f'The prediction file will be saved in {min(chunk_num, len(chromosome_name))} blocks.')

    chromosomes = list(zip(chromosome_name, chromosome_length))
    chromosomes.sort(key=lambda x: x[1], reverse=True)
    chromosomes_groups = [([], 0) for _ in range(chunk_num)]
    chromosomes_groups = [[list(chromosomes_group[0]), chromosomes_group[1]] for chromosomes_group in chromosomes_groups]
    for name, length in chromosomes:
        min_group = min(chromosomes_groups, key=lambda x: x[1])
        min_group[0].append(name)
        min_group[1] += length

    with open(output, 'w') as file:
        file.write('# This output was generated with ANNEVO (version 2.0.0).\n')
        file.write('# ANNEVO is a gene prediction tool written by YeLab.\n')
    seq_num = 1

    for chunk_order, (chromosome_name_list, _) in enumerate(chromosomes_groups):
        # for chunk_order, (chunk_start, chunk_end) in enumerate(chunk_index):
        print(f'Predicting chunk {chunk_order+1} / {chunk_num}')
        seq_id_chunk = chromosome_name_list
        seq_length_chunk = []
        windows_forward = []
        windows_reverse = []
        genome_predictions = {}
        offset = [0]
        count = 0
        for chromosome in seq_id_chunk:
            chromosome_seq_record = genome_seq[chromosome]
            sequence_forward = str(chromosome_seq_record.seq).upper()
            windows_reverse_disorder = []
            length = len(sequence_forward)
            seq_length_chunk.append(length)
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
                windows_forward.append(window_seq_forward)
                windows_reverse_disorder.append(reverse_complement(window_seq_forward))
                count += 1
            windows_reverse += windows_reverse_disorder[::-1]
            offset.append(count)

        predictions_forward = predict_probability(model, windows_forward, device, num_classes, batch_size, num_workers)
        predictions_reverse = predict_probability(model, windows_reverse, device, num_classes, batch_size, num_workers)

        for i, chromosome in enumerate(seq_id_chunk):
            length = seq_length_chunk[i]
            range_start = offset[i] * window_size
            range_end = offset[i + 1] * window_size
            predictions_forward_rec = predictions_forward[range_start:range_end][:length]
            predictions_reverse_rec = predictions_reverse[range_start:range_end][-length:]
            genome_predictions[chromosome] = [predictions_forward_rec, predictions_reverse_rec]

        print(f'Decoding chunk {chunk_order + 1} / {chunk_num}')
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
        with ProcessPoolExecutor(max_workers=threads) as executor:
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

        windows_forward.clear()
        windows_reverse.clear()
        genome_predictions.clear()
        torch.cuda.empty_cache()
        gc.collect()
