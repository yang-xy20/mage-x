#!/bin/sh
env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference
num_landmarks=20
num_agents=20
algo="rmappo"
exp="only_ctl_normalized_20agents_only_4maps"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} --seed ${seed} --n_training_threads 1 --n_rollout_threads 512 \
    --ctl_num_mini_batch 1 --exe_num_mini_batch 400 --episode_length 3 --num_env_steps 2000000000 \
    --ppo_epoch 3 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "mapping" \
    --user_name "yang-xy20" --use_macro --step_difference 2 --controller_num_agents 1 \
    --use_normalized
done

#--layer_N 2
