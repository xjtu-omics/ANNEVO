export PATH=/data/home/pyzhang/miniconda3/envs/ANNEVO/bin:$PATH
cd /data/home/pyzhang/ANNEVO

if [ -z "$1" ]; then
  echo "Usage: $0 <genome_path>"
  exit 1
fi
GENOME_PATH=$1

echo "------------------------Model prediction start------------------------"
start_time=$(date +%s)
python -m ANNEVO.prediction --genome $GENOME_PATH --lineage Embryophyta --model_prediction_path prediction_result/Arabidopsis_thaliana
end_time=$(date +%s)
execution_time_1=$((end_time - start_time))
echo "The model prediction of Arabidopsis took $execution_time_1 seconds."
echo "------------------------Model prediction end------------------------"
echo
echo
echo
echo "------------------------Gene decoding start------------------------"
start_time=$(date +%s)
python -m ANNEVO.decoding --genome $GENOME_PATH --model_prediction_path prediction_result/Arabidopsis_thaliana --output gff_result/Arabidopsis_thaliana_annotation.gff --cpu_num 48
end_time=$(date +%s)
execution_time_2=$((end_time - start_time))
echo "The gene decoding of Arabidopsis took $execution_time_2 seconds."
echo "------------------------Gene decoding end------------------------"
echo
echo
echo
execution_time_3=$((execution_time_1 + execution_time_2))
echo "The gene annotation of Arabidopsis took a total of ${execution_time_3} seconds."
echo "The number of predicted genes for Arabidopsis is $(awk '$3 == "transcript" {count++} END {print count+0}' gff_result/Arabidopsis_thaliana_annotation.gff)"
echo
echo
echo
echo "------------------------BUSCO Evaluation------------------------"

export PATH=/data/home/pyzhang/miniconda3/envs/busco/bin:$PATH
cd /data/home/pyzhang/busco

gffread -y Arabidopsis_protein.fasta -g /data/home/pyzhang/ANNEVO/example/Arabidopsis_thaliana/genome.fna /data/home/pyzhang/ANNEVO/gff_result/Arabidopsis_thaliana_annotation.gff
awk '/^>/{f=!d[$0];d[$0]=1}f' Arabidopsis_protein.fasta > tmp_Arabidopsis_protein.fasta
mv tmp_Arabidopsis_protein.fasta Arabidopsis_protein.fasta
sed '/^>/!s/\./X/g' Arabidopsis_protein.fasta > tmp_Arabidopsis_protein.fasta
mv tmp_Arabidopsis_protein.fasta Arabidopsis_protein.fasta
busco -i Arabidopsis_protein.fasta -c 48 -o "Arabidopsis_BUSCO" -m prot -l "busco_downloads/lineages/embryophyta_odb10" --offline -f
