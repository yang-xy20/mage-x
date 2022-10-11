#!/bin/sh
env="Drone"
scenario="Nav_Drone"  # simple_speaker_listener # simple_reference
num_agents=5
algo="rmappo"
exp="5agents_speed2_random_1.5_fix_obs_reward_more_info"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_drone.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 1024 \
    --exe_num_mini_batch 1 --ctl_num_mini_batch 1 --episode_length 121 --num_env_steps 2000000000 \
    --ppo_epoch 3 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "mapping" \
    --user_name "yang-xy20" --use_macro --freq 120 --step_difference 120 --controller_num_agents 1 \
    --use_centralized_V --agg_phy_steps 4 --epi_len_sec 4  --acce 5 --max_speed 2.00 --use_exe_gnn
done

#--obstacles