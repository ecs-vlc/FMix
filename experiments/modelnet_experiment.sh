
# Usage: `bash imagenet_experiment (msda_mode) (dataset path)`
# where msda_mode is one of [fmix, mixup, None]
# For multiple GPU, add --parallel=True

python ../trainer.py --dataset=modelnet --epoch=50 --schedule=10 20 30 40 --lr=0.001 --dataset-path=$2 --msda-mode=$1 --batch-size=16
