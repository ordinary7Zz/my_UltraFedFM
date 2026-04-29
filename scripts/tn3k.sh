cd "$(dirname "$0")/.." || exit 1

echo "结果将保存在 checkpoint 所在目录下自动生成的 eval_TN3K_<TIMESTAMP>/ 目录中"

DATASET='TN3K' # dataset name
CUDA_VISIBLE_DEVICES='0' python main_binary_segmentation.py \
    --datapath ./dataset/${DATASET}/ \
    --savepath ./output_dir/${DATASET} \
    --batch_size 16 \
    --note ${DATASET} \
    --resume ./output_dir/epoch_bestDice.pth \
    --eval 
