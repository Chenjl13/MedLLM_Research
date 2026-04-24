export CUDA_VISIBLE_DEVICES=1,2,3

###############################################
## Need To Set
###############################################
MODEL_PATH="pretrained_models/llava_med_v1.5"
LORA_CKP_NAME="finetune_MSLoRA-CR-ORTHO_3TASK-64-64_llava_med_v1.5"
DATASET_SPLITS=("slake-vqarad" "CXP" "iu-x-ray")

###############################################
## DO not care
###############################################
path_prefix="checkpoints/${LORA_CKP_NAME}"
previous_loras=()

for i in "${!DATASET_SPLITS[@]}"; do
    if [ "$i" -eq 0 ]; then
        path_suffix="${DATASET_SPLITS[$i]}"
        
    else
        path_suffix="${path_suffix}_${DATASET_SPLITS[$i]}"
    fi
    previous_loras+=("${path_prefix}/${path_suffix}/cl_lora.bin")
done

for i in "${!previous_loras[@]}"; do
    echo "Previous_Lora$((i+1)) = \"${previous_loras[$i]}\""
done

echo "========================================================"

TASK_MASK=1
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "previous loras: $previous_loras_string"
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"

TASK_MASK=2
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "previous loras: $previous_loras_string"
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"

TASK_MASK=3
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "previous loras: $previous_loras_string"
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"