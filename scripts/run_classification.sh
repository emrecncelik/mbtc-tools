INPUT_DIR="/home/emrecan/workspace/psychology-project/data"
OUTPUT_DIR="/home/emrecan/workspace/psychology-project/outputs"
CATEGORICAL_TARGETS=("attachment_label" "behavior_problems_label" "SSC_Tot" "Gen")
# AVERAGING=("simple" "weighted_average" "weighted_removal")
AVERAGING=("weighted_average" "simple")

for target in "${CATEGORICAL_TARGETS[@]}"
do
    for averaging in "${AVERAGING[@]}"
    do
        python classification.py \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR/$averaging" \
            --seed 1 \
            --averaging "$averaging" \
            --target_column "$target" \
            --weight_per_entity 5 \
            --min_weight 1 \
            --do_test
    done
done