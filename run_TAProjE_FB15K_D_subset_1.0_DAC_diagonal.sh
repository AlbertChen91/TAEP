CUDA_VISIBLE_DEVICES=0 python main.py --model_ TAProjE --data ./data/FB15K_D/ --save_dir /home/chenjf/TKBE/saved_model_TAProjE_FB15K_D_DAC_subset_0.10/TAProjE_saved_FB15K_D_dim_200_neg_0.5_len_60_subset_0.10_w_5_word_50_3000_DAC_diagonal_lr_0.001/ --resume 10 --batch 200 --test_batch 500 --dim 200 --epochs 100 --worker 5 --eval_per 1 --eval_start 0 --save_m 20 --optimizer adam --lr 0.001 --reg_weight 1e-5 --dropout 0.5 --neg_weight 0.5 --tolerance 5 --max_len 60 --encoder nbow --combine_methods dimensional_attentive --diagonal --subset 1.0 