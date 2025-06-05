#!/bin/bash
set -e
total_epoch=0

# Python script path
python_script=data_clean_main.py

for ((i=1; i<=$total_epoch; i++))
do
  torchrun --standalone --nproc_per_node=8 $python_script --actual_epoch $i --mode train
  torchrun --standalone --nproc_per_node=8 $python_script --actual_epoch $i --mode finetune
  torchrun --standalone --nproc_per_node=8 $python_script --actual_epoch $i --mode caption
done
