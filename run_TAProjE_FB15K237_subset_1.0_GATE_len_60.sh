CUDA_VISIBLE_DEVICES=0 python main.py --model_ TAProjE --data ./data/FB15K237/ --save_dir saved_model_TAProjE_FB15K_D_GATE_subset_0.5/ --resume 31 --batch 200 --test_batch 500 --dim 200 --epochs 100 --worker 5 --eval_per 1 --eval_start 0 --save_m 20 --optimizer adam --lr 0.01 --reg_weight 1e-5 --dropout 0.5 --neg_weight 0.5 --tolerance 5 --max_len 60 --encoder nbow --combine_methods gate --subset 0.5