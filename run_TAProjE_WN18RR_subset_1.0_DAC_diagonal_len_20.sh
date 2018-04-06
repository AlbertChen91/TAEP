CUDA_VISIBLE_DEVICES=1 python main.py --model_ TAProjE --data ./data/WN18RR/ --save_dir saved_model_TAProjE_WN18RR_DAC_subset_1.0/TAProjE_saved_WN18RR_dim_200_neg_0.5_len_20_subset_1.0_w_5_lr_0.001_DAC_diagonal/ --resume 23 --batch 200 --test_batch 500 --dim 200 --epochs 50 --worker 5 --eval_per 1 --eval_start 0 --save_m 40 --optimizer adam --lr 0.001 --reg_weight 1e-5 --dropout 0.5 --neg_weight 0.5 --tolerance 5 --max_len 20 --encoder nbow --combine_methods dimensional_attentive --diagonal --subset 1.0