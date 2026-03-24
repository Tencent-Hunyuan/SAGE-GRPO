export PYTHONPATH=`pwd`
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_DEBUG=DEBUG
#export NCCL_BLOCKING_WAIT=0     # 设置为0, 最大化计算与通信的重叠，提升训练效率
export NCCL_IB_TIMEOUT=22
#export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_TIMEOUT=600
export TOKENIZERS_PARALLELISM=false


echo "--------------------------------------------------------------------------"
#################################PARAMS setup#####################################
if [[ "${CURRENT_TIME}" = "" ]]; then
    CURRENT_TIME=$(date "+%Y.%m.%d-%H.%M.%S")
fi

if [[ "${START_EXPR_TIME}" = "" ]]; then
    START_EXPR_TIME=${CURRENT_TIME}
fi

output_dir=SAGE_GRPO_results/$START_EXPR_TIME

data_num_workers=8

# 检查当前的ckpt目录, 如果有ckpt, 则使用当前的ckpt; 如果没有, 则尝试使用配置
ckpt_cnt=`ls -d ${output_dir}/checkpoints/global_step-* | sort -t- -k2,2 -n -r | wc -l`
if [[ ${ckpt_cnt} -ne 0 ]]; then
    ckpt_dir=`cd ${output_dir}/checkpoints/; ls -d global_step-* | sort -t- -k2,2 -n -r | head -1`
    resume="${output_dir}/checkpoints/${ckpt_dir}"
    echo "there exists ckpt dir, set the resume=${ckpt_dir}"
else
    echo "there exists no ckpt dir, use config resume=${resume}"
fi

if [ -n "$resume" ] && [ "$resume" != "" ]; then
    training_params="$training_params --resume $resume"
fi

# Create logs directory
mkdir -p $output_dir/logs
node_rank=$INDEX
RANK_ID=${INDEX:-0}
CURRENT_LOG_FILE=$output_dir/logs/training_rank_${RANK_ID}.log
if [[ "${AUTO_RESUME_TRAIN}" = "true" ]]; then
    line_no=`grep -nw "${LOCAL_IP}" ${HOST_PATH} | cut -d: -f1`
    if [[ "${line_no}" != ""  ]]; then
        node_rank=$((${line_no} - 1))
        echo "get the local ip ${LOCAL_IP} node_rank ${node_rank} for resume train"
    else
        echo "fail to get the ${LOCAL_IP} node_rank for resume train!!!"
        exit 1
    fi
    CURRENT_LOG_FILE="$output_dir/logs/${CURRENT_TIME}_rank${node_rank}.log"
fi

# save the arguments to /dockerdata/.tccl/tccl.data for profiling
TCCL_DATA_DIR="/dockerdata/.tccl"
TCCL_DATA_FILE="${TCCL_DATA_DIR}/tccl.data"
rm -f "${TCCL_DATA_FILE}"
if [[ $node_rank = 0 ]]; then
    if [ -d "${TCCL_DATA_DIR}" ]; then
        echo "tccl directory ${TCCL_DATA_DIR} exists, use it!"
    else
        mkdir -p ${TCCL_DATA_DIR}
        echo "tccl directory ${TCCL_DATA_DIR} does not exist, create it!"
    fi

    if [[ "${AUTO_RESUME_TRAIN}" != "true" ]]; then
        echo "${NODE_IP_LIST}" | awk -F: -v RS=, '{gsub(/[0-9]+$/, ""); ips = ips ? ips "," : ""; ips = ips "\047" $1 "\047"} END{print "workers[" ips "]"}' >> ${TCCL_DATA_FILE}
    else
        awk 'BEGIN{printf "workers["} {printf "%s\047%s\047", (NR>1?",":""),$1} END{print "]"} ' ${HOST_PATH} >> ${TCCL_DATA_FILE}
    fi
    echo "--master_addr=${CHIEF_IP}" >> ${TCCL_DATA_FILE}
    echo "--master_port=37294" >> ${TCCL_DATA_FILE}
fi

# If config_file and config_index are provided, extract the corresponding config from the config file
# and save it to the output_dir
if [ -n "$config_file" ] && [ -n "$config_index" ]; then
python3 -c "
import yaml
import sys

config_file = sys.argv[1]
config_index = int(sys.argv[2])
output_file = sys.argv[3]

with open(config_file, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

if 'configs' in data and config_index < len(data['configs']):
    extracted_config = {'configs': [data['configs'][config_index]]}
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(extracted_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
else:
    print(f'Error: Config index {config_index} not found', file=sys.stderr)
    sys.exit(1)
" "$config_file" "$config_index" "$output_dir/$(basename $config_file)"
fi

torchrun --nnodes $HOST_NUM --nproc_per_node $HOST_GPU_NUM \
    --node_rank $node_rank \
    --rdzv_endpoint $CHIEF_IP:37375 \
    --rdzv_id 456 \
    post_train.py \
    --pretrained_model_root ./ckpts \
    --learning_rate 1e-5 \
    --batch_size 2 \
    --num_generations 4 \
    --max_steps 10000 \
    --output_dir ./outputs \
    --enable_fsdp \
    --enable_gradient_checkpointing \
    --sp_size 2 \
    --reward_checkpoint_mode "v3" \
    --validate_at_step0 False \
    --validate_video_length 121 \
    --validation_timestep_shift 5.0 \
    --use_grad_balancing True \
    --enable_timestep_permutation True \
    --sde_type "sage_grpo" \
    --kl_weight 1e-5 \
    --kl_coef 1e-7 \
    --use_moving_KL True \
    --update_ref_model_step 10 \
    --use_dual_kl True \
    --dual_kl_moving_weight 1.0 \
    --dual_kl_step_weight 0.1 \
    --reference_mode_offload True \
    2>&1 | tee ${CURRENT_LOG_FILE}

