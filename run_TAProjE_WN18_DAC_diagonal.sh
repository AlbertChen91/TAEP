python main.py --model_ TAProjE --data ./data/WN18/ --save_dir saved_model_TAProjE_WN18/TAProjE_saved_WN18_dim_200_neg_0.1_len_20_nbow_dimensional_attentive_diagonal/ --resume 29 --batch 200 --test_batch 500 --dim 200 --epochs 100 --worker 5 --eval_per 1 --eval_start 0 --save_m 20 --optimizer adam --lr 0.01 --clip 0.5 --reg_weight 1e-5 --dropout 0.5 --neg_weight 0.1 --tolerance 5 --max_len 20 --encoder nbow --combine_methods dimensional_attentive --diagonal