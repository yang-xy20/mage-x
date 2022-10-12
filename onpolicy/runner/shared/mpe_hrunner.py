import time
import numpy as np
import torch
from onpolicy.runner.shared.base_hierarchical_runner import HRunner
import wandb
import imageio
import torch.nn as nn

def _t2n(x):
    return x.detach().cpu().numpy()

class MPEHRunner(HRunner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(MPEHRunner, self).__init__(config)
        self.use_gnn = self.all_args.use_gnn
        self.use_exe_gnn = self.all_args.use_exe_gnn
        if self.use_gnn:
            self.init_ctl_input()
        if self.use_exe_gnn:
            self.init_exe_input()

    def run(self):
        self.envs.reset() 
        ctl_obs, ctl_rwd, ctl_done, ctl_info = self.envs.get_data("ctl")
        self.init_buffer(self.controller_buffer, self.controller_num_agents, ctl_obs, 'ctl')
        self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos = self.envs.get_data('exe')
        self.init_buffer(self.executor_buffer, self.executor_num_agents, self.exe_obs, 'exe')

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.controller_trainer.policy.lr_decay(episode, episodes)
                self.executor_trainer.policy.lr_decay(episode, episodes)
            ctl_step = 0
            exe_step = 0
            for step in range(self.episode_length):
                
                if step % (self.step_difference + 1) == 0:
                    # Sample actions
                    self.ctl_values, self.ctl_acts, self.ctl_act_log_probs, self.ctl_rnn_states, \
                    self.ctl_rnn_states_critic, self.ctl_actions_env = self.collect(ctl_step, 'ctl')
                    ctl_step += 1
                    self.envs.step(self.ctl_actions_env , 'ctl')

                elif step % (self.step_difference + 1) == self.step_difference:
                    # Sample actions
                    self.exe_values, self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, \
                        self.exe_rnn_states_critic, self.exe_actions_env = self.collect(exe_step, 'exe')
                    exe_step += 1
                    # Obser reward and next obs
                    self.envs.step(self.exe_actions_env, 'exe')
                    _, self.exe_rwds, self.exe_dones, self.exe_infos = self.envs.get_data('exe')
                    _, self.ctl_rwd, self.ctl_done, self.ctl_info = self.envs.get_data('ctl')#todo:done
                    self.exe_obs, _, _, _ = self.envs.get_data('exe')
                    self.ctl_obs, _, _, _ = self.envs.get_data('ctl')#todo:done
                    
                    exe_data = self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos, self.exe_values,\
                               self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, self.exe_rnn_states_critic
                    self.insert(exe_data, self.executor_buffer, self.executor_num_agents,'exe')
                    
                    # get controller datas from updated environment by executor step
                    # insert data into controller buffer
                    ctl_data = self.ctl_obs, self.ctl_rwd, self.ctl_done, self.ctl_info, self.ctl_values, \
                               self.ctl_acts, self.ctl_act_log_probs, self.ctl_rnn_states, self.ctl_rnn_states_critic
                    # insert data into buffer
                    self.insert(ctl_data, self.controller_buffer, self.controller_num_agents,'ctl')
                else:
                    # Sample actions
                    self.exe_values, self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, \
                    self.exe_rnn_states_critic, self.exe_actions_env = self.collect(exe_step, 'exe')
                    exe_step += 1
                    # execute executor step
                    self.envs.step(self.exe_actions_env, 'exe')

                    self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos = self.envs.get_data('exe')
                    
                    exe_data = self.exe_obs, self.exe_rwds, self.exe_dones, self.exe_infos, self.exe_values,\
                               self.exe_acts, self.exe_act_log_probs, self.exe_rnn_states, self.exe_rnn_states_critic
                    self.insert(exe_data, self.executor_buffer, self.executor_num_agents,'exe')

            # compute return and update network
            controller_train_infos = self.learn_update(self.controller_trainer, self.controller_buffer, 'ctl')
            executor_train_infos = self.learn_update(self.executor_trainer, self.executor_buffer, 'exe')
            
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
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

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        suc = []
                        for info in self.exe_infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                            if agent_id == 0 and "success_rate" in info[agent_id].keys():
                                suc.append(info[agent_id]['success_rate'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews
                        if agent_id == 0:
                            env_infos["success_rate"] = suc
                
                controller_train_infos["average_episode_controller_rewards"] = np.mean(self.controller_buffer.rewards) * (self.episode_length//self.step_difference)
                executor_train_infos["average_episode_executor_rewards"] = np.mean(self.executor_buffer.rewards) * self.episode_length
                print("controller average episode rewards is {}".format(controller_train_infos["average_episode_controller_rewards"]))
                print("executor average episode rewards is {}".format(executor_train_infos["average_episode_executor_rewards"]))
                self.log_train(controller_train_infos, total_num_steps)
                self.log_train(executor_train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
    
    def init_ctl_input(self):
        self.ctl_input= {}
        self.ctl_input['agent_pos'] = np.zeros((self.n_rollout_threads, self.controller_num_agents, self.num_agents, 2))
        self.ctl_input['land_pos'] = np.zeros((self.n_rollout_threads, self.controller_num_agents, self.num_agents, 2))
        self.ctl_input['rel_dis'] = np.zeros((self.n_rollout_threads, self.controller_num_agents, self.num_agents, self.num_agents, 1))
        self.ctl_share_input = self.ctl_input.copy()
    
    def init_exe_input(self):
        self.exe_input= {}
        self.exe_input['agent_state'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, 4))
        self.exe_input['target_goal'] = np.zeros((self.n_rollout_threads, self.num_agents, 1, 2))
        self.exe_input['other_pos'] = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents-1, 2))
        self.exe_share_input = self.exe_input.copy()

    def insert_ctl_data(self, obs):
        for e in range(self.n_rollout_threads):
            for a in range(self.controller_num_agents):
                for key in self.ctl_input.keys():
                    self.ctl_input[key][e, a] = obs[e, a][key]
        self.ctl_share_input = self.ctl_input.copy()

    def insert_exe_data(self, obs):
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                for key in self.exe_input.keys():
                    self.exe_input[key][e, a] = obs[e, a][key]
        self.exe_share_input = self.exe_input.copy()
            
    def init_buffer(self, buffer, num_agents, obs, mode):
        if self.use_centralized_V and mode =='exe':
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(num_agents, 1)
        else:
            share_obs = obs
        if mode == 'ctl' and self.use_gnn:
            self.insert_ctl_data(obs)
            for key in self.ctl_input.keys():
                buffer.obs[key][0] = self.ctl_input[key].copy()
            for key in self.ctl_share_input.keys():
                buffer.share_obs[key][0] = self.ctl_share_input[key].copy()
        elif mode == 'exe' and self.use_exe_gnn:
            self.insert_exe_data(obs)
            for key in self.exe_input.keys():
                buffer.obs[key][0] = self.exe_input[key].copy()
            for key in self.exe_share_input.keys():
                buffer.share_obs[key][0] = self.exe_share_input[key].copy()
        else:
            buffer.share_obs[0] = share_obs.copy()
            buffer.obs[0] = obs.copy()
    
    def learn_update(self, trainer, buffer, mode):
        # compute return and update network
        self.compute(trainer, buffer, mode)
        train_infos = self.train(trainer, buffer, mode)
        return train_infos

    @torch.no_grad()
    def collect(self, step, mode):
        action_space = None
        if mode == 'ctl':
            trainer = self.controller_trainer
            buffer = self.controller_buffer
            action_space = self.envs.ctl_action_space
            if self.use_gnn:
                concat_share_obs = {}
                concat_obs = {}
                for key in buffer.share_obs.keys():
                    concat_share_obs[key] = np.concatenate(buffer.share_obs[key][step])
                for key in buffer.obs.keys():
                    concat_obs[key] = np.concatenate(buffer.obs[key][step])
        elif mode == 'exe':
            action_space = self.envs.exe_action_space
            trainer = self.executor_trainer
            buffer = self.executor_buffer
            if self.use_exe_gnn:
                concat_share_obs = {}
                concat_obs = {}
                for key in buffer.share_obs.keys():
                    concat_share_obs[key] = np.concatenate(buffer.share_obs[key][step])
                for key in buffer.obs.keys():
                    concat_obs[key] = np.concatenate(buffer.obs[key][step])
        else:
            print('type of envs is wrong!')
            exit()
        trainer.prep_rollout() 
        if (mode == 'ctl' and  self.use_gnn) or (mode == 'exe' and  self.use_exe_gnn):
            value, action, action_log_prob, rnn_states, rnn_states_critic \
                = trainer.policy.get_actions(concat_share_obs,
                                concat_obs,
                                np.concatenate(buffer.rnn_states[step]),
                                np.concatenate(buffer.rnn_states_critic[step]),
                                np.concatenate(buffer.masks[step]))
        else:
            value, action, action_log_prob, rnn_states, rnn_states_critic \
                = trainer.policy.get_actions(np.concatenate(buffer.share_obs[step]),
                                np.concatenate(buffer.obs[step]),
                                np.concatenate(buffer.rnn_states[step]),
                                np.concatenate(buffer.rnn_states_critic[step]),
                                np.concatenate(buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
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
            action = action.detach().clone()
            action = nn.Sigmoid()(action)
            actions_env = np.array(np.split(_t2n(action), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data, buffer, num_agents, mode):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        
        if mode == 'ctl' and self.use_gnn:
            self.insert_ctl_data(obs)
            obs = self.ctl_input
            share_obs = self.ctl_share_input
        elif mode == 'exe' and self.use_exe_gnn:
            self.insert_exe_data(obs)
            obs = self.exe_input
            share_obs = self.exe_share_input
        else:
            if self.use_centralized_V and mode=='exe':
                share_obs = obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(num_agents, axis=1)
            else:
                share_obs = obs

        buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    def eval_act(self, trainer, obs, rnn_states, masks):
        trainer.prep_rollout()
        action, rnn_states = trainer.policy.act(np.concatenate(obs),
                                            np.concatenate(rnn_states),
                                            np.concatenate(masks),
                                            deterministic=True)
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

        if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(envs.action_space[0].shape):
                uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
        else:
            action = action.detach().clone()
            action = nn.Sigmoid()(action)
            actions_env = np.array(np.split(_t2n(action), self.n_rollout_threads))
        
        
        return action, actions_env, rnn_states

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            ctl_rnn_states = np.zeros((self.n_rollout_threads, self.controller_num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            ctl_masks = np.ones((self.n_rollout_threads, self.controller_num_agents, 1), dtype=np.float32)
            exe_rnn_states = np.zeros((self.n_rollout_threads, self.executor_num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            exe_masks = np.ones((self.n_rollout_threads, self.executor_num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()
                if step % (self.step_difference + 1) == 0:
                    ctl_obs, ctl_rwds, ctl_dones, ctl_infos = envs.get_data('ctl')
                    action, actions_env, ctl_rnn_states = self.eval_act(self.controller_trainer, ctl_obs, ctl_rnn_states, ctl_masks)
                    envs.step(actions_env,'ctl')
                    ctl_rnn_states[ctl_dones == True] = np.zeros(((ctl_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                    ctl_masks = np.ones((self.n_rollout_threads, self.controller_num_agents, 1), dtype=np.float32)
                else:
                    exe_obs, exe_rwds, exe_dones, exe_infos = envs.get_data('exe')
                    action, actions_env, exe_rnn_states = self.eval_act(self.executor_trainer, exe_obs, exe_rnn_states, exe_masks)
                    envs.step(actions_env,'exe')
                    exe_rnn_states[exe_dones == True] = np.zeros(((exe_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                    exe_masks = np.ones((self.n_rollout_threads, self.executor_um_agents, 1), dtype=np.float32)
                    exe_episode_rewards.append(exe_rwds)
                # Obser reward and next obs
                
                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average exe episode rewards is: " + str(np.mean(np.sum(np.array(exe_episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
