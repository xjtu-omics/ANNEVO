import argparse


def parse_gff_line(line):
    parts = line.strip().split('\t')
    if len(parts) < 9:
        return None
    attributes = parts[8]
    attr_dict = {key_value.split('=')[0]: key_value.split('=')[1] for key_value in attributes.split(';') if '=' in key_value}
    return {
        'seqid': parts[0],
        'source': parts[1],
        'type': parts[2],
        'start': parts[3],
        'end': parts[4],
        'score': parts[5],
        'strand': parts[6],
        'phase': parts[7],
        'attributes': attr_dict,
        'line': line.strip()
    }


def filter_gff(input_file, output_file):
    gene_dict = {}
    genes_to_remove = set()
    transcripts_to_remove = set()

    with open(input_file) as in_handle:
        lines = in_handle.readlines()

    for line in lines:
        if line.strip() and not line.startswith("#"):
            feature = parse_gff_line(line)
            if feature and feature['type'] == 'gene':
                gene_id = feature['attributes'].get('ID')
                if gene_id:
                    if gene_id in gene_dict:
                        genes_to_remove.add(gene_id)
                    else:
                        gene_dict[gene_id] = feature

    # if not genes_to_remove:
    #     return

    for line in lines:
        if line.strip() and not line.startswith("#"):
            feature = parse_gff_line(line)
            if feature:
                if feature['type'] == 'mRNA':
                    parent_id = feature['attributes'].get('Parent')
                    if parent_id in genes_to_remove:
                        transcripts_to_remove.add(feature['attributes'].get('ID'))

    with open(output_file, "w") as out_handle:
        for line in lines:
            if line.strip():
                if line.startswith("#"):
                    out_handle.write(line)
                else:
                    feature = parse_gff_line(line)
                    if feature:
                        gene_id = feature['attributes'].get('ID')
                        parent_id = feature['attributes'].get('Parent')
                        if feature['type'] == 'gene' and gene_id in genes_to_remove:
                            continue
                        if feature['type'] == 'mRNA' and gene_id in transcripts_to_remove:
                            continue
                        if feature['type'] in ['exon', 'CDS'] and parent_id in transcripts_to_remove:
                            continue
                    out_handle.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out genes with duplicate gene IDs in the annotations and their associated sub-features.")
    parser.add_argument("--input_file", help="Path to the input GFF file")
    parser.add_argument("--output_file", help="Path to the output GFF file")
    args = parser.parse_args()
    filter_gff(args.input_file, args.output_file)
