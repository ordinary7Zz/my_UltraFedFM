cd "$(dirname "$0")/.." || exit 1

DATASET='TN3K' # dataset name
CUDA_VISIBLE_DEVICES='0' python main_binary_segmentation.py \
    --datapath ./dataset/${DATASET}/ \
    --savepath ./output_dir/${DATASET} \
    --batch_size 16 \
    --note ${DATASET} \
    --resume ./output_dir/epoch_bestDice.pth \
    --eval 
