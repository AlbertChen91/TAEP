CUDA_VISIBLE_DEVICES=1 python main.py --model_ ProjE --data ./data/FB15K237/ --save_dir saved_model_ProjE_FB15K237_subset_1.0/ProjE_saved_FB15K237_dim_200_neg_0.5_subset_1.0_w_5/ --batch 200 --test_batch 500 --dim 200 --epochs 100 --worker 5 --eval_per 1 --eval_start 0 --save_m 20 --optimizer adam --lr 0.01 --reg_weight 1e-5 --dropout 0.5 --neg_weight 0.5 --tolerance 5 --subset 1.0