CUDA_VISIBLE_DEVICES=4 python /mnt/abu/evaluation/xin/c.py \
    --eval-dataset doc2dial \
    --data-folder /mnt/abu/data/ChatRAG-Bench/data \
    --query-encoder-path /mnt/abu/models/bge-large-en-v1.5 \
    --context-encoder-path /mnt/abu/models/bge-large-en-v1.5\
