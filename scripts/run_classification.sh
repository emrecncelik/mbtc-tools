INPUT_DIR="/home/emrecan/workspace/psychology-project/data"
OUTPUT_DIR="/home/emrecan/workspace/psychology-project/outputs"
CATEGORICAL_TARGETS=("attachment_label" "behavior_problems_label" "SSC_Tot" "Gen")
# AVERAGING=("simple" "weighted_average" "weighted_removal")
AVERAGING=("weighted_average")

for target in "${CATEGORICAL_TARGETS[@]}"
do
    for averaging in "${AVERAGING[@]}"
    do
        python classification.py \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --seed 1 \
            --averaging "$averaging" \
            --target_column "$target" 
    done
done