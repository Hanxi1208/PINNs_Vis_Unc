python train_uvp.py \
    --loss-weight 0.2 \
    --noise-level 0.0 --bias-level 0.3 --bias-type cos \
    --adam-steps 10000 --early-stop-patience 1000 \
    --lr-schedule none \
    --lbfgs-max-iter 10000 \
    --tag adam10k_lbfgs10k


