NODE_RANK=0  # 0 for the first node 0, 1 for the second node, and so on.
MASTER_ADDR="${cur_host/\.gadi\.nci\.org\.au/}"  # should be set as the ip of current node

# bash scripts/pretrain/vit_g_exp.sh $NODE_RANK $MASTER_ADDR
bash scripts/pretrain/vit_g_exp.sh