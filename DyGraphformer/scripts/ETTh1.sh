############################predict length 24####################################
python main_crossformer.py --data ETTh1 \
--in_len 168 --out_len 24 --seg_len 6 \
--learning_rate 1e-4 --itr 5 --gcn_dim 8

############################predict length 48####################################
python main_crossformer.py --data ETTh1 \
--in_len 168 --out_len 48 --seg_len 6 \
--learning_rate 1e-4 --itr 5 --gcn_dim 8

############################predict length 168###################################
python main_crossformer.py --data ETTh1  \
--in_len 720 --out_len 168 --seg_len 24 \
--learning_rate 1e-5 --itr 5 --gcn_dim 8

