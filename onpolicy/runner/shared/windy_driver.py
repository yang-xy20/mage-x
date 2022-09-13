#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import numpy as np
import torch
import wandb
from tmarl.drivers.shared_distributed.base_hierarchical_driver import Driver


def _t2n(x):
    return x.detach().cpu().numpy()


class WindyDriver(Driver):
    def __init__(self, config, client=None):
        super(WindyDriver, self).__init__(config, client=client)

    def inner_loop(self, episode, episodes, start):
        if self.boardcast_signal.check_stop():
            print('{} receive stop signal!'.format(self.program_type))
            self.should_run = False
        total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

        if not self.only_eval:
            # data collection
            self.actor_rollout()
            # model training
            controller_train_infos = self.learn_update(self.controller_trainer, self.controller_buffer, episode,
                                                       episodes)
            executor_train_infos = self.learn_update(self.executor_trainer, self.executor_buffer, episode, episodes)
            # post process
            self.post_process()
            # save models
            if episode % self.save_interval * 10 == 0 or episode == episodes - 1:
                self.save()
            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))
                if controller_train_infos:
                    controller_train_infos["FPS"] = int(total_num_steps / (end - start))
                if executor_train_infos:
                    executor_train_infos["FPS"] = int(total_num_steps / (end - start))

                self.log_train_info(controller_train_infos, executor_train_infos, total_num_steps)

        # eval
        self.evaluation(episode, total_num_steps)

    def actor_rollout(self):
        running_programs = ["actor", "whole", "local"]
        if self.program_type not in running_programs:
            return None

        self.controller_trainer.prep_rollout()
        self.executor_trainer.prep_rollout()

        if self.initialized_ctl_buffer is False:
            # Get updated data after the first reset
            ctl_obs, ctl_rwd, ctl_done, ctl_info, ctl_act_mask = self.envs.get_data('ctl')
            # init controller buffer
            self.init_buffer(self.controller_buffer, self.controller_num_agents, ctl_obs, ctl_act_mask)
            self.initialized_ctl_buffer = True

        ctl_step = 0
        exe_step = 0
        for step in range(self.episode_length):
            # In step 0, need to execute controller_step first,
            # and then initialize executor buffer according to the execution data
            if step % (self.envs.step_difference + 1) == 0:
                # Sample actions
                self.ctl_values, self.ctl_acts, self.ctl_act_log_probs, self.ctl_rnn_states, \
                self.ctl_rnn_states_critic, self.ctl_actions_env = self.collect(ctl_step, 'ctl')
                ctl_step += 1
                # execute controller step and update env (output executor's obs)
                self.envs.step(self.ctl_actions_env, 'ctl')

                # init executor buffer only once
                if self.initialized_exe_buffer is False:
                    # init executor buffer
                    self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos, self.exe_act_masks = \
                        self.envs.get_data('exe')

                    self.init_buffer(self.executor_buffer, self.executor_num_agents, self.exe_obs, self.exe_act_masks)
                    self.initialized_exe_buffer = True

                    self.exe_values, self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, \
                    self.exe_rnn_states_critic, self.exe_actions_env = self.collect(exe_step, 'exe')

                # get executor datas from updated environment by controller step
                self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos, \
                    self.exe_act_masks = self.envs.get_data('exe')

                # insert data into executor buffer
                exe_data = self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos, self.exe_act_masks, \
                           self.exe_values, self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, \
                           self.exe_rnn_states_critic
                self.insert(exe_data, self.executor_buffer, self.executor_num_agents)

            elif step % (self.envs.step_difference + 1) == self.envs.step_difference:
                # Sample actions
                self.exe_values, self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, \
                    self.exe_rnn_states_critic, self.exe_actions_env = self.collect(exe_step, 'exe')
                exe_step += 1
                # execute executor step
                self.envs.step(self.exe_actions_env, 'exe')

                # get controller datas from updated environment by executor step
                self.ctl_obs, self.ctl_rwd, self.ctl_done, self.ctl_info, self.ctl_act_mask = self.envs.get_data('ctl')

                # insert data into controller buffer
                ctl_data = self.ctl_obs, self.ctl_rwd, self.ctl_done, self.ctl_info, self.ctl_act_mask, \
                           self.ctl_values, self.ctl_acts, self.ctl_act_log_probs, self.ctl_rnn_states, \
                           self.ctl_rnn_states_critic
                self.insert(ctl_data, self.controller_buffer, self.controller_num_agents)
            else:
                # Sample actions
                self.exe_values, self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, \
                self.exe_rnn_states_critic, self.exe_actions_env = self.collect(exe_step, 'exe')

                exe_step += 1
                # execute executor step
                self.envs.step(self.exe_actions_env, 'exe')

                # get executor datas from updated environment
                self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos, \
                self.exe_act_masks = self.envs.get_data('exe')

                # insert data into executor buffer
                exe_data = self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos, self.exe_act_masks, \
                           self.exe_values, self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, \
                           self.exe_rnn_states_critic
                self.insert(exe_data, self.executor_buffer, self.executor_num_agents)

    def reset(self):
        if self.program_type not in ["actor", "whole", "local"]:
            return
        # reset envs
        self.envs.reset()
        self.initialized_ctl_buffer = False
        self.initialized_exe_buffer = False

    def init_buffer(self, buffer, num_agents, obs, act_mask):
        if self.concat_obs_as_share_obs:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(num_agents, 1)
        else:
            share_obs = obs
        buffer.init_buffer(share_obs.copy(), obs.copy(), act_mask.copy())

    @torch.no_grad()
    def collect(self, step, mode):
        action_space = None
        if mode == 'ctl':
            trainer = self.controller_trainer
            buffer = self.controller_buffer
            action_space = self.envs.ctl_action_space
        elif mode == 'exe':
            action_space = self.envs.exe_action_space
            trainer = self.executor_trainer
            buffer = self.executor_buffer
        else:
            print('type of envs is wrong!')
            exit()
        trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = trainer.algo_module.get_actions(
            np.concatenate(buffer.buffer.share_obs[step]),
            np.concatenate(buffer.buffer.obs[step]),
            np.concatenate(buffer.buffer.rnn_states[step]),
            np.concatenate(buffer.buffer.rnn_states_critic[step]),
            np.concatenate(buffer.buffer.masks[step]),
            np.concatenate(buffer.buffer.available_actions[step]))

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # rearrange action
        actions_env = self.rearrange_actions(action_space, actions)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def rearrange_actions(self, action_space, actions):
        actions_env = None
        if action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(action_space[0].shape):
                uc_actions_env = np.eye(action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError
        return actions_env

    def insert(self, data, buffer, num_agents):
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, \
        rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *buffer.buffer.rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.concat_obs_as_share_obs:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(num_agents, axis=1)
        else:
            share_obs = obs

        buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks,
                      available_actions=available_actions)

    def learn_update(self, trainer, buffer, episode, episodes):
        running_programs = ["server_learner", "learner", "whole", "local"]
        if self.program_type not in running_programs:
            return None

        # compute return and update network
        trainer.prep_training()

        if self.use_linear_lr_decay:
            trainer.algo_module.lr_decay(episode, episodes)

        self.compute(trainer, buffer)

        train_infos = self.train(trainer, buffer)
        return train_infos

    def post_process(self):
        if self.program_type in ["actor", "whole", "local"]:
            self.controller_buffer.after_update()
            self.executor_buffer.after_update()

    def log_train_info(self, controller_train_infos, executor_train_infos, total_num_steps):
        if self.program_type not in ["learner", "whole", "server_learner", "local"]:
            return
        controller_train_infos['controller_average_episode_rewards'] = np.mean(
            self.controller_buffer.buffer.rewards) * self.episode_length
        print("controller average episode rewards is {}".format(
            controller_train_infos["controller_average_episode_rewards"]))
        self.log_train(controller_train_infos, total_num_steps)
        executor_train_infos['executor_average_episode_rewards'] = np.mean(
            self.executor_buffer.buffer.rewards) * self.episode_length
        print("executor average episode rewards is {}".format(executor_train_infos["executor_average_episode_rewards"]))
        self.log_train(executor_train_infos, total_num_steps)

    def evaluation(self, episode, total_num_steps):
        if self.program_type in ["learner", "whole", "server_learner", "local"] or "evaluator" in self.program_type:
            if episode != 0 and episode % self.eval_interval == 0 and self.use_eval:
                self.controller_trainer.prep_rollout()
                self.executor_trainer.prep_rollout()
                self.eval(total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        # init eval goals
        num_done = 0
        eval_goals = np.zeros(self.all_args.eval_episodes)
        total_step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        # reset envs and init rnn and mask
        self.eval_envs.reset()
        self.eval_ctl_obs, self.eval_ctl_rwd, self.eval_ctl_done, self.eval_ctl_info, self.eval_ctl_act_masks = \
            self.eval_envs.get_data('ctl')
        self.eval_ctl_rnn_states = np.zeros((self.n_eval_rollout_threads, self.controller_num_agents, self.recurrent_N,
                                        self.hidden_size), dtype=np.float32)
        self.eval_ctl_masks = np.ones((self.n_eval_rollout_threads, self.controller_num_agents, 1), dtype=np.float32)

        self.eval_exe_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.executor_num_agents, self.recurrent_N,
             self.hidden_size), dtype=np.float32)
        self.eval_exe_masks = np.ones((self.n_eval_rollout_threads, self.executor_num_agents, 1),
                                      dtype=np.float32)

        self.controller_trainer.prep_rollout()
        self.executor_trainer.prep_rollout()

        step = 0
        # loop until enough episodes
        # while num_done < self.all_args.eval_episodes and step < self.episode_length:
        while num_done < self.all_args.eval_episodes:
            # # get actions
            # self.controller_trainer.prep_rollout()
            # self.executor_trainer.prep_rollout()

            if step % (self.eval_envs.step_difference + 1) == 0:
                self.eval_ctl_actions, self.eval_ctl_rnn_states = self.controller_trainer.algo_module.act(
                    np.concatenate(self.eval_ctl_obs),
                    np.concatenate(self.eval_ctl_rnn_states),
                    np.concatenate(self.eval_ctl_masks),
                    np.concatenate(self.eval_ctl_act_masks),
                    deterministic=True
                )

                self.eval_ctl_actions = np.array(np.split(_t2n(self.eval_ctl_actions), self.n_eval_rollout_threads))
                self.eval_ctl_rnn_states = np.array(np.split(_t2n(self.eval_ctl_rnn_states),
                                                             self.n_eval_rollout_threads))

                self.eval_ctl_actions_env = self.rearrange_actions(self.eval_envs.ctl_action_space,
                                                                   self.eval_ctl_actions)

                self.eval_envs.step(self.eval_ctl_actions_env, 'ctl')
                self.eval_exe_obs, self.eval_exe_rwds, self.eval_exe_dones, self.eval_exe_infos, \
                    self.eval_exe_act_masks = self.eval_envs.get_data('exe')
            elif step % (self.eval_envs.step_difference + 1) == self.eval_envs.step_difference:
                self.eval_exe_actions, self.eval_exe_rnn_states = self.executor_trainer.algo_module.act(
                    np.concatenate(self.eval_exe_obs),
                    np.concatenate(self.eval_exe_rnn_states),
                    np.concatenate(self.eval_exe_masks),
                    np.concatenate(self.eval_exe_act_masks),
                    deterministic=True
                )
                self.eval_exe_actions = np.array(np.split(_t2n(self.eval_exe_actions), self.n_eval_rollout_threads))
                self.eval_exe_rnn_states = np.array(np.split(_t2n(self.eval_exe_rnn_states),
                                                             self.n_eval_rollout_threads))
                self.eval_exe_actions_env = self.rearrange_actions(self.eval_envs.exe_action_space,
                                                                   self.eval_exe_actions)
                self.eval_envs.step(self.eval_exe_actions_env, 'exe')
                self.eval_ctl_obs, self.eval_ctl_rwd, self.eval_ctl_done, self.eval_ctl_info, self.eval_ctl_act_masks = \
                    self.eval_envs.get_data('ctl')

                # update goals if done
                eval_dones_env = np.all(self.eval_ctl_done, axis=-1)
                eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
                if np.any(eval_dones_unfinished_env):
                    for idx_env in range(self.n_eval_rollout_threads):
                        if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                            eval_goals[num_done] = self.eval_ctl_info[idx_env][0]["score_reward"]
                            num_done += 1
                            done_episodes_per_thread[idx_env] += 1
                unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

                # reset rnn and masks for done envs
                self.eval_ctl_rnn_states[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), self.controller_num_agents, self.recurrent_N, self.hidden_size),
                    dtype=np.float32)
                self.eval_ctl_masks = np.ones((self.n_eval_rollout_threads, self.controller_num_agents, 1),
                                              dtype=np.float32)
                self.eval_ctl_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(),
                                                                        self.controller_num_agents, 1),
                                                                       dtype=np.float32)
                total_step += 1
                if num_done == self.all_args.eval_episodes:
                    # get expected goal
                    eval_expected_goal = np.mean(eval_goals)

                    # log and print
                    print("-- total step: {}, eval cumulative reward is {}.".format(step, eval_expected_goal))
                    eval_env_infos = {}
                    eval_env_infos['eval_cumulative_reward'] = eval_goals
            else:
                self.eval_exe_actions, self.eval_exe_rnn_states = self.executor_trainer.algo_module.act(
                    np.concatenate(self.eval_exe_obs),
                    np.concatenate(self.eval_exe_rnn_states),
                    np.concatenate(self.eval_exe_masks),
                    np.concatenate(self.eval_exe_act_masks),
                    deterministic=True
                )
                self.eval_exe_actions = np.array(np.split(_t2n(self.eval_exe_actions), self.n_eval_rollout_threads))
                self.eval_exe_rnn_states = np.array(np.split(_t2n(self.eval_exe_rnn_states),
                                                             self.n_eval_rollout_threads))
                self.eval_exe_actions_env = self.rearrange_actions(self.eval_envs.exe_action_space,
                                                                   self.eval_exe_actions)
                self.eval_envs.step(self.eval_exe_actions_env, 'exe')
                # get executor datas from updated environment
                self.eval_exe_obs, self.eval_exe_rwds, self.eval_exe_dones, self.eval_exe_infos, \
                self.eval_exe_act_masks = self.eval_envs.get_data('exe')

            step += 1