ip_with_mask=$NODE_ADDR
export HOST_IP=${ip_with_mask%/*}
export VLLM_HOST_IP=$HOST_IP
RAY_START_CMD="ray start"
if [ "${NODE_RANK}" == "0" ]; then
    RAY_START_CMD+=" --node-ip-address $HOST_IP --head --port=6379 --block &"
    eval "$RAY_START_CMD"
    sleep 60
    python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME --dtype bfloat16 --trust-remote-code --tensor-parallel-size 8 --pipeline-parallel-size 2 --max-model-len 28000 --port 80 --gpu-memory-utilization 0.95 --distributed-executor-backend ray
else
    RAY_START_CMD+=" --node-ip-address $HOST_IP --address=${PRIMARY_ADDR}:6379 --block"
    eval "$RAY_START_CMD"
fi
