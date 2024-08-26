source /data/zhongz2/anaconda3/bin/activate th23
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0



python -m serve_controller --host 0.0.0.0 --port 10000
python -m serve_gradio_web_server --controller http://localhost:10000 --model-list-mode reload
# llava_llama_3_1
python -m serve_model_worker --host 0.0.0.0 --controller-address http://localhost:10000 \
    --port 40000 --worker http://localhost:40000

# llava_qwen_2
python -m serve_model_worker --host 0.0.0.0 --controller-address http://localhost:10000 \
    --port 40001 --worker http://localhost:40001 \
    --model_path "/Users/zhongz2/down/finetune_qwen_2" \
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

