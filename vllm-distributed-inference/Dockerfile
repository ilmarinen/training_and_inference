FROM runpod/worker-v1-vllm:v2.7.0stable-cuda12.1.0

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ray

COPY start_ray_cluster.sh .

# Enable HF Hub Transfer
# ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Expose port 80
EXPOSE 80

ENTRYPOINT ["bash", "start_ray_cluster.sh"]
