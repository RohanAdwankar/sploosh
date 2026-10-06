#!/bin/bash
# One time setup on a fresh VM: venv, CPU torch, the model's config and tokenizer, the shard index.
set -e
cd "$(dirname "$0")"
python3 -m venv /home/user/venv
/home/user/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/home/user/venv/bin/pip install -q safetensors numpy huggingface_hub requests tiktoken transformers blobfile
mkdir -p /home/user/k2 /home/user/expcache
for f in config.json configuration_deepseek.py tokenization_kimi.py tokenizer_config.json tiktoken.model; do
  curl -sSL -o /home/user/k2/$f https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/main/$f
done
curl -sSL -o /home/user/k2/index.json https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/main/model.safetensors.index.json
echo "setup done"
