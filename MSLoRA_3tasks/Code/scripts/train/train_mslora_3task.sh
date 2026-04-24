export WANDB_MODE=disabled
#export CUDA_VISIBLE_DEVICES=1

PORT=29501
echo "Generated master port: $PORT"

TRAIN_VERSION="finetune_MSLoRA-CR-ORTHO_3TASK"
PRETRAINED_MODEL_PATH="pretrained_models/llava_med_v1.5"
MODEL_NAME=$(basename "$PRETRAINED_MODEL_PATH")

LR=2e-4
BATCH_SIZE=4
GRADIENT_ACC_STEPS=4
LORA_RANK=64
LORA_ALPHA=64

MAX_TASK=1
IMAGE_FOLDER="data/Slake-VQARad"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/slake-vqarad"

EPOCH=1

deepspeed --include localhost:1,2,3 --master_port $PORT llava/train/train.py \
    --deepspeed scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb

################################
# Task2: CXP
################################

MAX_TASK=2
IMAGE_FOLDER="data/CXP"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"

pretrain_lora_path1="checkpoints/finetune_MSLoRA-CR-ORTHO_3TASK-64-64_llava_med_v1.5/slake-vqarad/cl_lora.bin"

OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/slake-vqarad_CXP"

EPOCH=1

deepspeed --include localhost:1,2,3 --master_port $PORT llava/train/train.py \
    --deepspeed scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --previous_lora_path $pretrain_lora_path1 \
    --is_same_modality 1 \
    --seed 42 --alpha 0.01 --beta 0.1 \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb

################################
# Task3: IU-X-Ray
################################

MAX_TASK=3
IMAGE_FOLDER="data/IU-X-Ray"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"

pretrain_lora_path1="checkpoints/finetune_MSLoRA-CR-ORTHO_3TASK-64-64_llava_med_v1.5/slake-vqarad/cl_lora.bin"
pretrain_lora_path2="checkpoints/finetune_MSLoRA-CR-ORTHO_3TASK-64-64_llava_med_v1.5/slake-vqarad_CXP/cl_lora.bin"

OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/slake-vqarad_CXP_iu-x-ray"

EPOCH=1

deepspeed --include localhost:1,2,3 --master_port $PORT llava/train/train.py \
    --deepspeed scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --previous_lora_path $pretrain_lora_path1 $pretrain_lora_path2 \
    --is_same_modality 1 1 \
    --seed 42 --alpha 0.01 --beta 0.1 \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb