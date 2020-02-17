
# Usage: `bash imagenet_experiment (msda_mode) (dataset path)`
# where msda_mode is one of [fmix, mixup, None]
# For multiple GPU, add --parallel=True

python ../trainer.py --dataset=tinyimagenet --epoch=200 --schedule 150 180 --batch-size=128 --lr=0.1 --dataset-path=$2 --msda-mode=$1
