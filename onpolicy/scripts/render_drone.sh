#!/bin/sh
env="Drone"
scenario="Nav_Drone"  # simple_speaker_listener # simple_reference
num_agents=1
algo="rmappo"
exp="debug_speed_xy_0.1_mini1_len20"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python render/render_drone.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --render_episodes 4 \
    --exe_num_mini_batch 1 --ctl_num_mini_batch 1 --episode_length 1201 --num_env_steps 2000000000 --ppo_epoch 3 --use_ReLU --gain 0.01 \
    --lr 7e-4 --critic_lr 7e-4 --wandb_name "mapping" --user_name "yang-xy20" --use_macro \
    --step_difference 1200 --controller_num_agents 1 --use_centralized_V --agg_phy_steps 4 \
    --epi_len_sec 20  --acce 3 --max_speed 1.00 --use_wandb --model_dir "./results/Drone/Nav_Drone/rmappo/debug_xy_0.1_mini1_len20/wandb/run-20221002_043442-3k14of35/files"
done

#--obstacles