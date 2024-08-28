source /data/zhongz2/anaconda3/bin/activate th24
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0


# cli
MODEL_PATH=/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/meta-llama/Meta-Llama-3.1-8B-Instruct/google/siglip-so400m-patch14-384/llama_3_1/finetune/checkpoint-500/
# MODEL_PATH=/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/finetune_llama_3_1_without_pretrain_conch/checkpoint-1800
MODEL_PATH=/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/meta-llama/Meta-Llama-3.1-8B-Instruct/openai/clip-vit-large-patch14-336/llama_3_1/finetune
MODEL_NAME=llava_llama_3_1
CONV_VERSION=llama_3_1
# MODEL_PATH=/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/Qwen/Qwen2-7B/openai/clip-vit-large-patch14-336/qwen_2/finetune/checkpoint-300/
# MODEL_NAME=llava_qwen_2
# CONV_VERSION=qwen_2
python serve_cli.py \
--model_path ${MODEL_PATH} \
--model_name ${MODEL_NAME} \
--conv_version ${CONV_VERSION} \
--device "cuda" \
--attn_implementation "flash_attention_2" \
--temperature 0.2 --top_p 0.8 --top_k 30 --debug --max-new-tokens 512 \
--image-file ./examples/figure-002-a71770_large.jpg


python -m serve_controller --host 0.0.0.0 --port 10000
python -m serve_gradio_web_server --controller http://localhost:10000 --model-list-mode reload
# llava_llama_3_1
python -m serve_model_worker --host 0.0.0.0 --controller-address http://localhost:10000 \
    --port 40000 --worker http://localhost:40000

# llava_qwen_2
python -m serve_model_worker --host 0.0.0.0 --controller-address http://localhost:10000 \
    --port 40001 --worker http://localhost:40001 \
    --model_path "/Users/zhongz2/down/checkpoint-300" \
    --model_name "llava_qwen_2" \
    --conv_version "qwen_2" \
    --cache_dir "/Users/zhongz2/down/cache_dir" \
    --attn_implementation "sdpa" \
    --device "mps"

# Server
python serve_cli.py \
    --model_path "/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/finetune_llama_3_1_with_pretrain" \
    --model_name "llava_llama_3_1" \
    --conv_version "llama_3_1" \
    --cache_dir "/data/zhongz2/data/cache_dir/" \
    --attn_implementation "flash_attention_2" \
    --device "cuda"
# Mac
python serve_cli.py \
    --model_path "/Users/zhongz2/down/finetune_llama_3_1_with_pretrain" \
    --model_name "llava_llama_3_1" \
    --conv_version "llama_3_1" \
    --cache_dir "/Users/zhongz2/down/cache_dir" \
    --attn_implementation "eager" \
    --device "mps"

# Server
python serve_cli.py \
    --model_path "/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/finetune_qwen_2" \
    --model_name "llava_qwen_2" \
    --conv_version "qwen_2" \
    --cache_dir "/data/zhongz2/data/cache_dir/" \
    --attn_implementation "flash_attention_2" \
    --device "cuda" \
    --temperature 0.1
python serve_cli.py \
    --model_path "/data/zhongz2/temp29/output_llava_llama_3/pretrain_anyres_debug3/finetune_qwen_2" \
    --model_name "llava_qwen_2" \
    --conv_version "qwen_2" \
    --cache_dir "/data/zhongz2/data/cache_dir/" \
    --attn_implementation "flash_attention_2" \
    --device "cuda" \
    --max-new-tokens 512 --temperature 0.9 --repetition_penalty 1.05 --top_p 0.8 --top_k 20

# Mac
python serve_cli.py \
    --model_path "/Users/zhongz2/down/finetune_qwen_2" \
    --model_name "llava_qwen_2" \
    --conv_version "qwen_2" \
    --cache_dir "/Users/zhongz2/down/cache_dir" \
    --attn_implementation "eager" \
    --device "mps" \
    --temperature 0.1










