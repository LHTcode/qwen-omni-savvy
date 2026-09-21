#!/bin/bash
# Qwen2.5-Omni seven-channel spatial-audio evaluation for SAVVY-Bench.

set -e

cp models/qwen2_5_omni_spatial_audio_FULL.py third_party/lmms_eval/lmms_eval/models/qwen2_5_omni_spatial_audio_FULL.py
cp -r tasks/spatial_avqa third_party/lmms_eval/lmms_eval/tasks/

# Versions recommended by the Qwen2.5-Omni project. FlashAttention-2 requires
# a compatible NVIDIA GPU, CUDA toolkit, and an existing PyTorch installation.
# pip install "transformers==4.52.3" accelerate
# pip install qwen-omni-utils[decord] -U
# # Match PyTorch to the host CUDA 11.8 toolkit before compiling flash-attn.
# python -m pip install "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" \
#     --index-url https://download.pytorch.org/whl/cu118
# python -m pip install -U flash-attn --no-build-isolation
# pip install "flash-attn==2.7.4.post1" --no-build-isolation

pip install "setuptools==78.1.0" audioread
git config --global --add safe.directory /home/fds-admin/user_files/lihanting/MLLM_Research/savvy
cd third_party/lmms_eval
pip install -e .
cd ../../

# Local Qwen2.5-Omni checkpoint. Use an absolute path because `~` is not
# expanded inside lmms-eval's comma-separated --model_args string.
model_path="$PWD/models/Qwen"
export model_path
if [ ! -f "$model_path/config.json" ]; then
    echo "Missing local Qwen checkpoint: $model_path/config.json"
    exit 1
fi

# Link the processed video and its seven-channel spatial WAV into the layout
# expected by spatial_avqa_doc_to_visual() and qwen2_5_omni_spatial_audio_FULL.py.
aea_processed_dir="$PWD/../SAVVY/data_utils/aea/aea_processed"
video_dir="$PWD/data/spatial_avqa/videos"
mkdir -p "$video_dir"
for seq_dir in "$aea_processed_dir"/*; do
    seq=$(basename "$seq_dir")
    video="$seq_dir/video_merged/$seq.mp4"
    audio="$seq_dir/audio_spatial/$seq.wav"
    if [ -f "$video" ] && [ -f "$audio" ]; then
        ln -sf "$(realpath "$video")" "$video_dir/$seq.mp4"
        ln -sf "$(realpath "$audio")" "$video_dir/$seq.wav"
    fi
done

CUDA_LAUNCH_BLOCKING=1 bash scripts/eval_model_base.sh \
    --model qwen2_5_omni_7b_spatial_FULL \
    --num_processes 1 \
    --benchmark spatial_avqa \
    --output_path logs/qwen2_5_omni_7b_spatial_FULL \
    --limit "${LIMIT:-}"
