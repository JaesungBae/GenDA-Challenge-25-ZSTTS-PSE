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
            -p 'speecht5_synth_50utt' 
        done
    done
done