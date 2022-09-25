#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference
num_landmarks=30
num_agents=30
algo="rmappo"
exp="no_joint_has_entropy_macro_30agents"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 256 \
    --num_mini_batch 10 --episode_length 61 --num_env_steps 2000000000 --ppo_epoch 3 --use_ReLU --gain 0.01 \
    --lr 7e-4 --critic_lr 7e-4 --wandb_name "zero-shot-transfer" --user_name "yang-xy20" --use_macro \
    --step_difference 60 --controller_num_agents 1
done
