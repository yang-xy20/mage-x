import time
import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
import math

# from ...transmit import Client
from onpolicy.utils.shared_buffer import SharedReplayBuffer



def _t2n(x):
    return x.detach().cpu().numpy()


class HRunner(object):
    def __init__(self, config):
        self.all_args = config['all_args']

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.controller_num_agents = self.all_args.controller_num_agents
        self.executor_num_agents = self.num_agents

        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

        # print(self.all_args)

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        #self.concat_obs_as_share_obs = self.all_args.concat_obs_as_share_obs
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        # In HRL, num_env_steps represents the controller's step numbers
        self.num_env_steps = self.all_args.num_env_steps
        self.step_difference = self.all_args.step_difference
        # In HRL, num_env_steps represents the controller's episode length
        # self.episode_length = self.all_args.episode_length
        if self.all_args.episode_length % (self.all_args.step_difference + 1) == 0:
            self.episode_length = self.all_args.episode_length
        else:
            self.episode_length = self.all_args.episode_length + (
                self.all_args.step_difference + 1 - (
                    self.all_args.episode_length % (self.all_args.step_difference + 1)
                )
            )
            print('self.episode_length: ', self.episode_length)
            assert self.all_args.episode_length % (self.all_args.step_difference + 1) != 0, print(
                'self.all_args.episode_length is not times step_difference + 1')

        self.n_rollout_threads = self.all_args.n_rollout_threads
        #self.learner_n_rollout_threads = self.all_args.learner_n_rollout_threads
        # self.local_n_rollout_threads = self.all_args.local_n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        #self.use_single_network = self.all_args.use_single_network
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        #self.only_eval = self.all_args.only_eval
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # # reverb address
        # self.server_address = self.all_args.server_address

        # self.distributed_type = self.all_args.distributed_type

        # self.actor_num = self.all_args.actor_num
        # self.program_type = self.all_args.program_type

        # if self.distributed_type == 'async' and self.program_type == 'whole':
        #     print('can\'t use async mode when program_type is whole!')
        #     exit()

        # if self.program_type in ["whole", "local"]:
        #     assert self.actor_num == 1, "when running actor and learner the same time, the actor number should be 1, " \
        #                                 "but received {}".format(self.actor_num)
        # dir
        self.model_dir = self.all_args.model_dir

        #if self.program_type not in ["actor"]:
        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        # if "mappo" in self.algorithm_name:
        #     if self.use_single_network:
        #         from tmarl.algorithms.r_mappo_single_dis.mappo_algorithm import MAPPOAlgorithm as TrainAlgo
        #         from tmarl.algorithms.r_mappo_single_dis.mappo_module import MAPPOModule as AlgoModule
        #     else:
        #         from tmarl.algorithms.r_mappo_distributed.mappo_algorithm import MAPPOAlgorithm as TrainAlgo
        #         if "transformer" in self.algorithm_name:
        #             from tmarl.algorithms.r_mappo_distributed.transformer_module import TransformerModule as AlgoModule
        #         else:
        #             from tmarl.algorithms.r_mappo_distributed.mappo_module import MAPPOModule as AlgoModule
        # elif "mappg" in self.algorithm_name:
        #     if self.use_single_network:
        #         from tmarl.algorithms.r_mappg_single.r_mappg_single import R_MAPPG as TrainAlgo
        #         from tmarl.algorithms.r_mappg_single.algorithm.rMAPPGPolicy import R_MAPPGPolicy as AlgoModule
        #     else:
        #         from tmarl.algorithms.r_mappg.r_mappg import R_MAPPG as TrainAlgo
        #         from tmarl.algorithms.r_mappg.algorithm.rMAPPGPolicy import R_MAPPGPolicy as AlgoModule
        # else:
        #     raise NotImplementedError
        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        # policy network
        self.controller_algo_module = Policy(self.all_args, self.envs.ctl_observation_space[0],
                                                 self.envs.ctl_share_observation_space[0],
                                                 self.envs.ctl_action_space[0], use_macro=True ,device=self.device)

        self.executor_algo_module = Policy(self.all_args, self.envs.exe_observation_space[0],
                                               self.envs.exe_share_observation_space[0],
                                               self.envs.exe_action_space[0], use_macro=False, device=self.device)

        # algorithm
        self.controller_trainer = TrainAlgo(self.all_args, self.controller_algo_module, device=self.device)
        self.executor_trainer = TrainAlgo(self.all_args, self.executor_algo_module, device=self.device)

        if self.model_dir is not None:
            self.restore()

        # Distributed correlation
        # if self.program_type not in ['local']:
        #     self.controller_model_keys, self.executor_model_keys = self.get_model_keys()

        # self.step_difference = self.all_args.step_difference if hasattr(self.all_args, 'step_difference') else 1

        # self.signal_client = None
        # self.weight_client = None
        # self.data_client = None
        # self.init_clients(client)

        # buffer
        ctl_episode_length = self.episode_length // (self.all_args.step_difference + 1)
        print('ctl_episode_length: ', ctl_episode_length)
        exe_episode_length = self.episode_length - ctl_episode_length
        print('exe_episode_length: ', exe_episode_length)

        self.controller_buffer = SharedReplayBuffer(self.all_args,
                                        self.controller_num_agents,
                                        self.envs.ctl_observation_space[0],
                                        self.envs.ctl_share_observation_space[0],
                                        self.envs.ctl_action_space[0],
                                        episode_length=ctl_episode_length)

        self.executor_buffer = SharedReplayBuffer(self.all_args,
                                                  self.executor_num_agents,
                                                  self.envs.exe_observation_space[0],
                                                  self.envs.exe_share_observation_space[0],
                                                  self.envs.exe_action_space[0],
                                                  episode_length=exe_episode_length)

        self.initialized_ctl_buffer = None
        self.initialized_exe_buffer = None

    def run(self):
        start = time.time()
        # todo 这里没有判断
        episodes = int(self.num_env_steps) // self.episode_length // self.learner_n_rollout_threads

        self.reset()

        if self.program_type == 'actor' and self.distributed_type == 'async':
            self.should_run = True
            episode = 0
            while self.should_run:
                self.inner_loop(episode, episodes, start)
                episode += 1
        else:
            for episode in range(episodes):
                self.inner_loop(episode, episodes, start)

        if self.use_wandb:
            running_programs = ["learner", "server_learner", "local", "whole"]
            if self.all_args.program_type in running_programs:
                wandb.finish()

        self.boardcast_signal.send_stop(stop_message='{} sending stop signal!'.format(self.program_type))
        time.sleep(5)

    def warmup(self):
        raise NotImplementedError

    def collect(self, step, trainer, buffer, envs):
        raise NotImplementedError

    def insert(self, data, buffer, num_agents):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self, trainer, buffer, mode):
        trainer.prep_rollout()

        if 'transformer' in self.algorithm_name:
            next_values = trainer.policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                                         np.concatenate(buffer.obs[-1]),
                                                         np.concatenate(buffer.rnn_states_critic[-1]),
                                                         np.concatenate(buffer.masks[-1]))
        elif self.use_gnn and mode == 'ctl':
            concat_share_obs = {}
            for key in buffer.share_obs.keys():
                concat_share_obs[key] = np.concatenate(buffer.share_obs[key][-1])
            next_values = trainer.policy.get_values(concat_share_obs,
                                                         np.concatenate(buffer.rnn_states_critic[-1]),
                                                         np.concatenate(buffer.masks[-1]))

        else:
            next_values = trainer.policy.get_values(np.concatenate(buffer.share_obs[-1]),
                                                         np.concatenate(buffer.rnn_states_critic[-1]),
                                                         np.concatenate(buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        buffer.compute_returns(next_values, trainer.value_normalizer)

    def train(self, trainer, buffer, mode):
        trainer.prep_training()
        train_infos = trainer.train(buffer, mode)
        buffer.after_update()
        return train_infos

    def save(self):
        ctl_policy_actor = self.controller_trainer.policy.actor
        torch.save(ctl_policy_actor.state_dict(), str(self.save_dir) + "/ctl_actor.pt")
        ctl_policy_critic = self.controller_trainer.policy.critic
        torch.save(ctl_policy_critic.state_dict(), str(self.save_dir) + "/ctl_critic.pt")

        exe_policy_actor = self.executor_trainer.policy.actor
        torch.save(exe_policy_actor.state_dict(), str(self.save_dir) + "/exe_actor.pt")
        exe_policy_critic = self.executor_trainer.policy.critic
        torch.save(exe_policy_critic.state_dict(), str(self.save_dir) + "/exe_critic.pt")
        # running_programs = ["local", "learner", "server_learner", "whole"]
        # if not self.program_type in running_programs:
        #     return
        # controller_models = self.controller_trainer.algo_module.models
        # executor_models = self.executor_trainer.algo_module.models

        # for model_name in controller_models:
        #     torch.save(controller_models[model_name].state_dict(),
        #                str(self.save_dir) + "/controller_{}.pt".format(model_name))

        # for model_name in executor_models:
        #     torch.save(executor_models[model_name].state_dict(),
        #                str(self.save_dir) + "/executor_{}.pt".format(model_name))

    def restore(self):
        controller_models = self.controller_trainer.policy
        executor_models = self.executor_trainer.policy
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/ctl_actor.pt')
        controller_models.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/ctl_critic.pt')
            controller_models.critic.load_state_dict(policy_critic_state_dict)
        
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/exe_actor.pt')
        executor_models.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/exe_critic.pt')
            executor_models.critic.load_state_dict(policy_critic_state_dict)
 
        # print('restore model from {}'.format(str(self.model_dir)))
        # controller_models = self.controller_trainer.algo_module.models
        # executor_models = self.executor_trainer.algo_module.models

        # for model_name in controller_models:
        #     state_dict = torch.load(str(self.model_dir) + '/controller_{}.pt'.format(model_name),
        #                             map_location=self.device)
        #     controller_models[model_name].load_state_dict(state_dict)

        # for model_name in executor_models:
        #     state_dict = torch.load(str(self.model_dir) + '/executor_{}.pt'.format(model_name),
        #                             map_location=self.device)
        #     controller_models[model_name].load_state_dict(state_dict)

    # def init_clients(self, client=None):
    #     pass_program_types = ["local"]
    #     if self.program_type not in pass_program_types:
    #         self.signal_client = client or Client(self.server_address)
    #         self.weight_client = client or Client(self.server_address)
    #         self.data_client = client or Client(self.server_address)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
