
# Usage: `bash imagenet_experiment (model) (msda_mode) (dataset path)`
# where:
# model is one of [resnet, wrn, densenet]
# msda_mode is one of [fmix, mixup, None]
# For multiple GPU, add --parallel=True

if [ "$1" == "resnet" ]
then
    model=ResNet18
    epoch=200
    schedule=(100 150)
    bs=128
fi

if [ "$1" == "wrn" ]
then
    model=wrn
    epoch=300
    schedule=(100 150 225)
    bs=32
fi

if [ "$1" == "densenet" ]
then
    model=DenseNet190
    epoch=300
    schedule=(100 150 225)
    bs=32
fi

python ../trainer.py --dataset=fashion --model=$model --epoch=$epoch --schedule=$schedule --lr=0.1 --dataset-path=$3 --msda-mode=$2 --batch-size=$bs
