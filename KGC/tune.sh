#!/usr/bin/env bash

for COEF in 1.00E-08 1.00E-09 1.00E-06
do
  for LR in 0.1 0.05 0.01 0.005 0.001
  do
    for BZ in 100 500 1000 5000 10000
    do
      for NEG in 10 20 50 100 200 500
      do
        for DIM in 16 32 64 128
        do
          dglke_train --model_name DistMult \
          --dataset family --data_path data/family \
          --data_files train.txt valid.txt test.txt \
          --format raw_udd_hrt \
          --batch_size $BZ \
          --neg_sample_size $NEG \
          --hidden_dim $DIM \
          --lr $LR \
          --max_step 1000 \
          --log_interval 500 \
          --batch_size_eval 256 \
          --test \
          --regularization_coef $COEF \
          --gpu 0
          echo "--regularization_coef $COEF --lr $LR --hidden_dim $DIM --neg_sample_size $NEG --batch_size $BZ"
          echo
          echo
          echo
          echo
        done
      done
    done
  done
done
