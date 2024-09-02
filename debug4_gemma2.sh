#!/bin/bash

echo "run job2"
echo `date`


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    MYTMP_DIR=/tmp/zhongz2
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
fi

JSON_FOLDER="${MYTMP_DIR}/train_json"
IMAGE_FOLDER="${MYTMP_DIR}"
VIDEO_FOLDER="${MYTMP_DIR}"

PRETRAIN_DATA="${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/llava_med_alignment_500k_cleaned.json"
FINETUNE_DATA="${JSON_FOLDER}/llava_med_instruct_60k_cleaned.json ${JSON_FOLDER}/la_tune_256k.json ${JSON_FOLDER}/lrv_tune_331k.json ${JSON_FOLDER}/lvis_tune_220k_.json ${JSON_FOLDER}/svit_tune_157k.json ${JSON_FOLDER}/nlp_tune.json"
FINETUNE_DATA="${JSON_FOLDER}/llava_med_instruct_60k_cleaned.json ${JSON_FOLDER}/llava_image_tune_cleaned.json"
save_steps=100
num_train_epochs=1


per_device_train_batch_size=2
gradient_accumulation_steps=8
learning_rate=2e-5
data_type_str="--bf16 True --tf32 True"
deepspeed_config="zero3_deepspeed"
num_workers=4

conv_version=plain

conv_version=llama_3
model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
pretrain_ckpt_path=     # "--pretrain_ckpt_path /data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/mm_projector.bin"
output_dir=/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/finetune2_test2

conv_version=gemma_2
model_name_or_path="google/gemma-2-9b-it"
vision_tower_name_or_path=google/siglip-so400m-patch14-384
vision_tower_name_or_path=openai/clip-vit-large-patch14-336
pretrain_ckpt_path=
output_dir=/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/${model_name_or_path}/${vision_tower_name_or_path}/${conv_version}/finetune
MASTER_PORT=25205
per_device_train_batch_size=2
gradient_accumulation_steps=8

if [ ! -d ${output_dir} ]; then mkdir -p ${output_dir}; fi

if [ "${SLURM_JOB_NODELIST}" != "" ]; then
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    NNODES=$SLURM_NNODES
    GPUS_PER_NODE=2
else
    MASTER_ADDR=`hostname`
    NNODES=1
    GPUS_PER_NODE=2
fi
MASTER_PORT=25199

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    main.py \
    --deepspeed ${deepspeed_config}.json \
    --model_name_or_path ${model_name_or_path} \
    --vision_tower_name_or_path ${vision_tower_name_or_path} \
    --data_path $FINETUNE_DATA \
    --image_folder $IMAGE_FOLDER \
    ${data_type_str} \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 1 \
    --mm_projector_lr 2e-5 \
    --mm_vision_tower_lr 2e-6 \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --dataloader_num_workers ${num_workers} \
    --report_to tensorboard \
    --cache_dir /data/zhongz2/data/cache_dir \
    --dataloader_drop_last True \
    --group_by_modality_length True \
    --gradient_checkpointing True \
    --image_aspect_ratio anyres \
    --model_max_length 8192 \
    --conv_version ${conv_version} \
    ${pretrain_ckpt_path} \
    2>&1 | tee log_debug3_finetune_${conv_version}_fixed.txt

exit;
--log_level debug \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --deepspeed ./scripts/${deepspeed_config}.json \

    --pretrain_ckpt_path ${pretrain_ckpt_path} \



# 20240713
vim /data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/transformers/trainer.py 
vim /data/zhongz2/anaconda3/envs/th21_ds0144/lib/python3.11/site-packages/transformers/trainer.py 
add the following to "_maybe_log_save_evaluate" function 
```
            # self.accelerator.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()
            if self.is_deepspeed_enabled:
                self.deepspeed.empty_partition_cache()
```

# zip from a list of files
zip archive.zip -@ < sample_images/samples.txt

rustup install 1.75.0
rustup toolchain list
rustup default 1.75*******



vim /data/zhongz2/anaconda3/envs/th21_ds0144/lib/python3.11/site-packages/transformers/utils/import_utils.py












