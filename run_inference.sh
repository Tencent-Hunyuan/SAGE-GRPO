# export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
# export T2V_REWRITE_MODEL_NAME="<your_model_name>"
# export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
# export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='A girl holding a paper with words "Hello, world!"'
VALID_VIDEO_CSV=assets/demo_sample.csv

GLOBAL_SEED=930
SEED=$GLOBAL_SEED
ASPECT_RATIO=16:9
FIXED_SIZE=480x864
RESOLUTION=480p
NUM_SAMPLE_STEPS=40
OUTPUT_PATH=./outputs/output_samples_864x480_step40_shift5_gs1_seed$GLOBAL_SEED
MODEL_PATH=./ckpts # Path to pretrained model

# Configuration for faster inference
N_INFERENCE_GPU=8 # Parallel inference GPU count
CFG_DISTILLED=false # Inference with CFG distilled model, 2x speedup
SAGE_ATTN=false # Inference with SageAttention
SPARSE_ATTN=false # Inference with sparse attention (only 720p models are equipped with sparse attention). Please ensure flex-block-attn is installed
OVERLAP_GROUP_OFFLOADING=false # Only valid when group offloading is enabled, significantly increases CPU memory usage but speeds up inference
ENABLE_CACHE=false # Enable feature cache during inference. Significantly speeds up inference.
CACHE_TYPE=deepcache # Support: deepcache, teacache, taylorcache
ENABLE_STEP_DISTILL=false # Enable step distilled model for 480p I2V, recommended 8 or 12 steps, up to 6x speedup
OFFLOADING=false # Enable offloading

# Configuration for better quality
REWRITE=false # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution


torchrun --nproc_per_node=$N_INFERENCE_GPU generate.py \
  --prompt "$PROMPT" \
  --valid_video_csv "$VALID_VIDEO_CSV" \
  --resolution $RESOLUTION \
  --fixed_size $FIXED_SIZE \
  --aspect_ratio $ASPECT_RATIO \
  --seed $SEED \
  --num_inference_steps $NUM_SAMPLE_STEPS \
  --offloading $OFFLOADING \
  --rewrite $REWRITE \
  --cfg_distilled $CFG_DISTILLED \
  --enable_step_distill $ENABLE_STEP_DISTILL \
  --sparse_attn $SPARSE_ATTN --use_sageattn $SAGE_ATTN \
  --enable_cache $ENABLE_CACHE --cache_type $CACHE_TYPE \
  --overlap_group_offloading $OVERLAP_GROUP_OFFLOADING \
  --sr $ENABLE_SR --save_pre_sr_video \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH