# /usr/bin/env bash
python3 train.py --max_len 128 \
                  --batch_size 32 \
                  --num_epochs 50 \
                  --lr 1e-4 \
                  --dr_rate 0 \
                  --hidden_size 768 \
                  --num_classes 5 \
                  --seed 42 \
                  --verbose True \
                  --save True \
                  --device cuda:0
