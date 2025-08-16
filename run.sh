# gowalla
nohup python3 -u engine.py --dataset_name gowalla --epochs 500 --bpr_batch 5120 --test_batch 5120 --latent_dim 256 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-gowalla-256-rank.pth.tar --seed 47 --device_id 0 > 241223_gowalla.out 2>&1 &
nohup python3 -u engine.py --dataset_name gowalla --epochs 500 --bpr_batch 5120 --test_batch 5120 --latent_dim 128 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-gowalla-128-rank.pth.tar --seed 47 --device_id 0 > 250104_gowalla_128dim.out 2>&1 &
nohup python3 -u engine.py --dataset_name gowalla --epochs 1 --bpr_batch 5120 --test_batch 5120 --latent_dim 128 --seed 47 --device_id 0 > 241226_gowalla_128dim_full.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 -u engine.py --dataset_name gowalla --epochs 500 --bpr_batch 20000 --test_batch 20000 --latent_dim 64 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-gowalla-64-rank.pth.tar --seed 47 --device_id 0 > log/gowalla/gmpq_64/250312.out 2>&1 &

# amazon-book
nohup python3 -u engine.py --dataset_name amazon-book --epochs 500 --bpr_batch 5120 --test_batch 5120 --latent_dim 256 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-book-256-rank.pth.tar --seed 47 --device_id 1 > 241223_amazon-book.out 2>&1 &
nohup python3 -u engine.py --dataset_name amazon-book --epochs 500 --bpr_batch 5120 --test_batch 5120 --latent_dim 128 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-book-128-rank.pth.tar --seed 47 --device_id 0 > 241224_amazon-book_128dim.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 -u engine.py --dataset_name amazon-book --epochs 500 --bpr_batch 10000 --test_batch 10000 --latent_dim 256 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-book-256-rank.pth.tar --seed 47 --device_id 0 > log/amazon-book/gmpq_256/250312.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python3 -u engine.py --dataset_name amazon-book --epochs 500 --bpr_batch 10000 --test_batch 10000 --latent_dim 64 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-book-64-rank.pth.tar --seed 47 --device_id 0 > log/amazon-book/gmpq_64/250312.out 2>&1 &


# yelp2020
nohup python3 -u engine.py --dataset_name yelp2020 --epochs 500 --bpr_batch 10240 --test_batch 10240 --latent_dim 256 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-yelp2020-256-rank.pth.tar --seed 47 --device_id 2 > log/yelp2020/gmpq_256/250308.out 2>&1 &

nohup python3 -u engine.py --dataset_name yelp2020 --epochs 500 --bpr_batch 10240 --test_batch 10240 --latent_dim 128 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-yelp2020-128-rank.pth.tar --seed 47 --device_id 0 > log/yelp2020/gmpq_128/250308.log 2>&1 &


# 消融实验 1
CUDA_VISIBLE_DEVICES=1 nohup python3 -u engine.py --dataset_name yelp2020 --epochs 500 --bpr_batch 10240 --test_batch 10240 --latent_dim 128 --additional_alias ROS消融 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-yelp2020-128-rank.pth.tar --seed 47 --device_id 0 > log/yelp2020/gmpq_128_ROS/250318.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 -u engine.py --dataset_name gowalla --epochs 500 --bpr_batch 10240 --test_batch 10240 --latent_dim 128 --additional_alias ROS消融 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-gowalla-128-rank.pth.tar --seed 47 --device_id 0 > log/gowalla/gmpq_128_ROS/250318.log 2>&1 &
# 消融实验 2
CUDA_VISIBLE_DEVICES=1 nohup python3 -u engine.py --dataset_name yelp2020 --epochs 500 --bpr_batch 10240 --test_batch 10240 --latent_dim 128 --additional_alias 梯度反传消融 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-yelp2020-128-rank.pth.tar --seed 47 --device_id 0 > log/yelp2020/gmpq_128_梯度反传消融/250319.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 -u engine.py --dataset_name gowalla --epochs 500 --bpr_batch 10240 --test_batch 10240 --latent_dim 128 --additional_alias 梯度反传消融 --use_pretrain_init --pretrain_state_dict checkpoints/bgr-gowalla-128-rank.pth.tar --seed 47 --device_id 0 > log/gowalla/gmpq_128_梯度反传消融/250319.log 2>&1 &
