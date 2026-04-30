# Re-train ANNEVO
When you need to incorporate additional species or retrain ANNEVO on a specific clade, you can follow the scripts below:  
```bash
train_species_list="The species list used for training model"
val_species_list="The species list used for validating model"
h5_data_path="The path to store h5 file" 
mkdir -p tmp

# The file must be cleared before each run.
rm -f ${h5_data_path}/train.h5 ${h5_data_path}/train_with_intergenic.h5
rm -f ${h5_data_path}/val.h5 ${h5_data_path}/val_with_intergenic.h5

for species_name in "${train_species_list[@]}"; do
    path_to_genome="The path to species genome"
    path_to_annotation="The path to species annotation"
    # Filter out duplicated gene IDs and other issues that may cause parsing errors in the Biopython package
    python src/filter_wrong_record.py --input_file ${path_to_annotation} --output_file "tmp/tmp_${species_name}.gff"
    # Convert the genome sequence and annotation into H5 data for model training.
    python generate_datasets.py --genome ${path_to_genome} --annotation "tmp/tmp_${species_name}.gff" --output_file "${h5_data_path}/train" --threads 64
    rm -f "tmp/tmp_${species_name}.gff"
done
for species_name in "${val_species_list[@]}"; do
    path_to_genome="The path to species genome"
    path_to_annotation="The path to species annotation"
    python src/filter_wrong_record.py --input_file ${path_to_annotation} --output_file "tmp/tmp_${species_name}.gff"
    python generate_datasets.py --genome ${path_to_genome} --annotation "tmp/tmp_${species_name}.gff" --output_file "${h5_data_path}/val" --threads 64
    rm -f "tmp/tmp_${species_name}.gff"
done

# Train the deep learning model
python model_train.py --h5_path ${h5_data_path} --model_save_path path_to_new_model.pt
```

# Fine tuning
In cases where closely related species are limited or unavailable for the target genome, one of ANNEVO’s five main trained models can be selected as a starting point for fine-tuning.
```bash
# Filter out duplicated gene IDs and other issues that may cause parsing errors in the Biopython package
fine_tune_species_list="The species list used for fine tuning model"
h5_data_path="The path to store h5 file"
mkdir -p tmp

# The file must be cleared before each run.
rm -f ${h5_data_path}/fine_tune.h5 ${h5_data_path}/fine_tune_with_intergenic.h5

for species_name in "${fine_tune_species_list[@]}"; do
    path_to_genome="The path to species genome"
    path_to_annotation="The path to species annotation"
    python src/filter_wrong_record.py --input_file ${path_to_annotation} --output_file "tmp/tmp_${species_name}.gff"
    python generate_datasets.py --genome ${path_to_genome} --annotation "tmp/tmp_${species_name}.gff" --output_file "${h5_data_path}/fine_tune" --threads 64
    rm -f "tmp/tmp_${species_name}.gff"
done

# Fine tuning deep learning model
python fine_tune.py --model_path path_to_existing_model.pt --model_save_path path_to_new_model.pt --h5_path ${h5_data_path}
```