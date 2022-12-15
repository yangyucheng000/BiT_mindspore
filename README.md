# BiT_mindspore
Use mindspore framework to implement BiT method.

## Training
First run get_dataset.sh to download Cifar10 dataset, and then you can use the following command to train the model:
```
nohup ./dist_train.sh 8 --name dist_152x4 --model BiT-M-R152x4 --logdir ./bit_logs --dataset cifar10 --datadir /home/ma-user/work/big_transfer/data/cifar-10-batches-bin --base_lr 0.001 --batch 16 --eval_every 100 --num_classes 10 > train.log 2>&1 &
```
