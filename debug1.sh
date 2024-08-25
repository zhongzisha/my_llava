#!/bin/bash

ACTION=${1}

set -x -e
echo "run job1 on `hostname`"

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    MYTMP_DIR=/tmp/zhongz2
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    DATA_ROOT=/data/zhongz2/data/LLaVA-Med/video
fi
cd $MYTMP_DIR
cp -r ${DATA_ROOT}/MoE/train_json $MYTMP_DIR/
# cp ${DATA_ROOT}/all*.jpg ${DATA_ROOT}/all*.mp4 $MYTMP_DIR/


unzip -qq ${DATA_ROOT}/MoE/llava_med_all.zip
unzip -qq ${DATA_ROOT}/MoE/llava_image.zip

unzip -qq ${DATA_ROOT}/MoE/llava_image_tune.zip
# unzip -qq ${DATA_ROOT}/MoE/lvis_tune.zip
# unzip -qq ${DATA_ROOT}/MoE/lrv_tune.zip
# unzip -qq ${DATA_ROOT}/MoE/svit_tune.zip
# mkdir mimicit_tune && cd mimicit_tune
# unzip -qq ${DATA_ROOT}/MoE/mimicit_tune/LA.zip
cd $MYTMP_DIR
if [ "$CLUSTER_NAME" == "FRCE" ]; then 
    unzip -qq ${DATA_ROOT}/finetune_data_LLaVA-Med.zip
else 
    unzip -qq ${DATA_ROOT}/../finetune_data_LLaVA-Med.zip
fi

if [ "$ACTION" == "eval" ]; then
    # cp ${DATA_ROOT}/MoE/llava_image_tune_cleaned.json train_json/
    if [ "$CLUSTER_NAME" == "FRCE" ]; then
        # frce
        mkdir eval
        cd $MYTMP_DIR/eval && unzip -qq ${DATA_ROOT}/MoE/eval/eval.zip
        # cd $MYTMP_DIR/eval/vqav2 && unzip -qq ${DATA_ROOT}/MoE/eval/vqav2/test2015.zip
        cd $MYTMP_DIR/eval/textvqa && unzip -qq ${DATA_ROOT}/MoE/eval/textvqa/train_val_images.zip
        cd $MYTMP_DIR/eval/textvqa && cp ${DATA_ROOT}/MoE//eval/textvqa/TextVQA_0.5.1_val.json .
        cd $MYTMP_DIR
    else
        mkdir eval
        cd $MYTMP_DIR/eval && unzip -qq ${DATA_ROOT}/eval/eval.zip
        # cd $MYTMP_DIR/eval/vqav2 && unzip -qq ${DATA_ROOT}/eval/vqav2/test2015.zip
        cd $MYTMP_DIR/eval/textvqa && unzip -qq ${DATA_ROOT}/eval/textvqa/train_val_images.zip
        cd $MYTMP_DIR/eval/textvqa && cp ${DATA_ROOT}/eval/textvqa/TextVQA_0.5.1_val.json .
        cd $MYTMP_DIR
    fi
fi











