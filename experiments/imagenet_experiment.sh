

# Usage: `bash imagenet_experiment (msda_mode) (dataset path)`
# where msda_mode is one of [fmix, mixup, None]
# For multiple GPU, add --parallel=True

python ../trainer.py --dataset=imagenet --epoch=90 --model=torch_resnet101 --schedule 30 60 80 --batch-size=256 --lr=0.4 --lr-warmup=True --dataset-path=$2 --msda-mode=$1
