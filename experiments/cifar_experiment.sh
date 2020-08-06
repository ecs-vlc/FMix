
# Usage: `bash imagenet_experiment (cifar) (model) (msda_mode) (dataset path)`
# where:
# cifar is one of [cifar10, cifar100]
# model is one of [resnet, wrn, densenet, pyramidnet]
# msda_mode is one of [fmix, mixup, None]
# For multiple GPU, add --parallel=True

if [ "$1" == "cifar10" ]
then
    ds=cifar10
fi
if [ "$1" == "cifar100" ]
then
    ds=cifar10
fi

if [ "$2" == "resnet" ]
then
    model=ResNet18
    epoch=200
    schedule=(100 150)
    bs=128
    cosine=False
fi

if [ "$2" == "wrn" ]
then
    model=wrn
    epoch=200
    schedule=(100 150)
    bs=128
    cosine=False
fi

if [ "$2" == "densenet" ]
then
    model=DenseNet190
    epoch=300
    schedule=(100 150 225)
    bs=32
    cosine=False
fi

if [ "$2" == "pyramidnet" ]
then
    model=aa_PyramidNet
    epoch=1800
    schedule=2000
    bs=64
    cosine=True
fi

python ../trainer.py --dataset=$ds --model=$model --epoch=$epoch --schedule=$schedule --lr=0.1 --dataset-path=$4 --msda-mode=$3 --batch-size=$bs --cosine-scheduler=$cosine
