#!/bin/bash

speakers=('F0')
sizes=('medium' 'small' 'tiny')
rates=(0.0001)


for spk in ${speakers[@]}; do
    for s in ${sizes[@]}; do
        for r in ${rates[@]}; do
            echo "spk: $spk, size: $s, rate: $r"
            python train.py \
            -s $spk \
            -r $r \
            -i $s \
            -p 'speecht5_synth_50utt' \
            --save_path results/test \
            --train_data_csv_dir examples/csv_files \
            --min_epoch 0 \
            --max_epoch 1001 \
            --batch_size 8
        done
    done
done