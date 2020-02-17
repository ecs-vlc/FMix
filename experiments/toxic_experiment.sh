
# Usage: `bash imagenet_experiment (msda_mode) (dataset path)`
# where msda_mode is one of [fmix, mixup, None]
# For multiple GPU, add --parallel=True

python ../trainer.py --dataset=toxic --epoch=10 --batch-size=64 --lr=1e-4 --dataset-path=$2 --msda-mode=$1
