CUDA_VISIBLE_DEVICES=1 python main_r.py --model_ TAProjE_R --data ./data/FB15K237/ --save_dir saved_model_TAProjE_R_GATE/TAProjE_R_FB15K237_dim_200_neg_0.5_len_60_w_5_word_50_3000_gate_lr_0.01/ --batch 200 --test_batch 500 --dim 200 --epochs 100 --worker 5 --eval_per 1 --eval_start 0 --save_m 20 --optimizer adam --lr 0.01 --reg_weight 1e-5 --dropout 0.5 --neg_weight 0.5 --tolerance 10 --max_len 60 --encoder nbow --combine_methods gate