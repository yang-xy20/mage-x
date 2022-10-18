#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference
num_landmarks=5
num_agents=5
algo="rmappo"
exp="ours_5agents"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed 2 --n_training_threads 1 --n_rollout_threads 128 \
    --ctl_num_mini_batch 1 --exe_num_mini_batch 5 --episode_length 61 --num_env_steps 2000000000 \
    --ppo_epoch 3 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "mapping" \
    --user_name "yang-xy20" --use_macro --step_difference 60 --controller_num_agents 1 \
    --use_normalized --use_exe_gnn --hidden_size 32
done
#--layer_N 2
