dataset='oucuavseg'
split='1_2'

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch \
	--nproc_per_node=$2 \
	--master_addr=localhost \
	--master_port=$3 \
	visualization.py \
	--labeled-id-path splits/oucuavseg/1_2/labeled.txt \
	--unlabeled-id-path splits/oucuavseg/1_2/unlabeled.txt \
	--save-path exp/oucuavseg/allspark/1_2_cledgeerror
