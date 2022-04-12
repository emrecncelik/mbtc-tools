CATEGORICAL_TARGETS=("attachment_label" "behavior_problems_label" "SSC_Tot" "Gen")

for target in "${CATEGORICAL_TARGETS[@]}"
do
    python format_data.py \
        --mst_dir /home/emrecan/workspace/psychology-project/data/MST_all \
        --mst_var_file /home/emrecan/workspace/psychology-project/data/mst_variables.csv \
        --attachment_dir /home/emrecan/workspace/psychology-project/data/Attachment \
        --behavior_dir "/home/emrecan/workspace/psychology-project/data/Behavior Problems" \
        --output_dir "/home/emrecan/workspace/psychology-project/data" \
        --target_variable "$target" \
        --target_variable_type categorical \
        --test_size 0.15 \
        --sep "<#>"
done