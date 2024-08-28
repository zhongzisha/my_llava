
if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th21_ds
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    MYTMP_DIR=/tmp/zhongz2
else
    source /data/zhongz2/anaconda3/bin/activate th23
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    MYTMP_DIR=/lscratch/$SLURM_JOB_ID
fi
export PYTHONPATH=`pwd`:$PYTHONPATH
