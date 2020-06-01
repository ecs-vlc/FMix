
# Usage: `bash imagenet_experiment (msda_mode) (dataset path) (alpha)`
# where msda_mode is one of [fmix, mixup, None]
# For multiple GPU, add --parallel=True

python ../trainer.py --dataset=commands --epoch=90 --schedule 30 60 80 --lr=0.01 --dataset-path=$2 --msda-mode=$1 --alpha=$3
