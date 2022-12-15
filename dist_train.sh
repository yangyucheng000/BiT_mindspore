#!/usr/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh RANK_SIZE [optional arguments]"
echo "For example: bash run.sh 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
RANK_SIZE=$1

EXEC_PATH=$(pwd)

# test_dist_8pcs()
# {
#     export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
#     export RANK_SIZE=8
# }

# test_dist_2pcs()
# {
#     export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_2pcs.json
#     export RANK_SIZE=2
# }

# test_dist_${RANK_SIZE}pcs

export PYTHONPATH="$EXEC_PATH":$PYTHONPATH

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./train_ms_graph_dist.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./train_ms_graph_dist.py ${@:2} > train.log$i 2>&1 &
    cd ../
done
rm -rf device0
mkdir device0
cp ./train_ms_graph_dist.py ./device0
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python ./train_ms_graph_dist.py ${@:2} > train.log0 2>&1
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../
