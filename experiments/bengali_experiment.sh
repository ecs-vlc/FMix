
# Usage: `bash imagenet_experiment (msda_mode) (type) (dataset path)`
# where msda_mode is one of [fmix, mixup, None]
# type is one of [r, c, v]
# For multiple GPU, add --parallel=True

python ../trainer.py --dataset bengali_${2} --fold 0 --model se_resnext50_32x4d --epoch 100 --schedule 50 75 --batch-size 512 --lr=0.1 --dataset-path=$3 --msda-mode=$1
