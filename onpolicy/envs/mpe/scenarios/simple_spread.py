import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.name = 'spread'
        world.use_gnn = args.use_gnn
        world.use_exe_gnn = args.use_exe_gnn
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks  # 3
        world.collaborative = True
        world.pred_goal_id = np.zeros((world.num_agents))
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.id = i
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.id = i
            landmark.id = i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        self.prev_agent_state = []
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3, +3, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            self.prev_agent_state.append(agent.state.p_pos.copy())
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-3, +3, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        self.goal_id, self.gt_dists = self.compute_macro_allocation(world)
        
    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world, mode):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        
        rew = 0
        cover = 0
        if mode == 'exe':
            goal_id = world.pred_goal_id[agent.id]
            target_goal = world.landmarks[int(goal_id)]
            dists = np.sqrt(np.sum(np.square(agent.state.p_pos - target_goal.state.p_pos)))
            rew -= dists
            if dists <= world.agents[0].size + world.landmarks[0].size:
                rew += 1
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
        elif mode == 'ctl':
            dists = 0
            if not np.any(world.pred_goal_id):
                rew = 0
            else:
                for a in world.agents:
                    goal_id = world.pred_goal_id[a.id]
                    target_goal = world.landmarks[int(goal_id)]
                    dists += np.sqrt(np.sum(np.square(self.prev_agent_state[a.id] - target_goal.state.p_pos)))
                rew = -(dists - self.gt_dists)
                # self.prev_agent_state = []
                # for a in world.agents:
                #     self.prev_agent_state.append(a.state.p_pos.copy())
                # for l in world.landmarks:
                #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                #             for a in world.agents]
                #     rew -= min(dists)
                #     if min(dists) <= world.agents[0].size + world.landmarks[0].size:
                #         cover += 1
                #         # give bonus for cover landmarks
                #         rew += 1
                # # success bonus
                # if cover == len(world.landmarks):
                #     rew += 4 * len(world.landmarks) 
        
        return rew

    def observation(self, agent, world, mode):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_gt_pos = []
        agent_pos = []
        comm = []
        other_pos = []
        other_gt_pos = []
        rel_pos = np.zeros((world.num_agents, world.num_landmarks, 1))
        goal_id = world.pred_goal_id[agent.id]
        for land_id, entity in enumerate(world.landmarks):  # world.entities:
            if int(goal_id) == int(land_id):
                target_gt_goal = entity.state.p_pos
                target_goal = entity.state.p_pos - agent.state.p_pos
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_gt_pos.append(entity.state.p_pos)
            for agent_id, other in enumerate(world.agents):
                rel_pos[agent_id, land_id] = np.sqrt(np.sum(np.square(entity.state.p_pos - other.state.p_pos)))
                if int(goal_id) == int(land_id):
                    agent_pos.append(other.state.p_pos)
                    if other is agent:
                        continue
                    comm.append(other.state.c)
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_gt_pos.append(other.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents        
        id_vector = np.zeros(len(world.agents))
        id_vector[agent.id] = 1
        if mode == 'exe':
            if world.use_exe_gnn:
                info = {}
                info['agent_state'] = np.concatenate([agent.state.p_vel]+[agent.state.p_pos])
                info['target_goal'] = np.stack([target_gt_goal])
                info['other_pos'] = np.stack(other_gt_pos)
                # info['agent_id'] = np.stack(id_vector)
                return info
            else:
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [target_goal] + other_pos + [id_vector])
            #entity_pos + other_pos + [id_vector] + comm)
        elif mode == 'ctl':
            if world.use_gnn:
                info = {}
                info['agent_pos'] = np.stack(agent_pos)
                info['land_pos'] = np.stack(entity_gt_pos)
                info['rel_dis'] = rel_pos
                return info
            else:
                return np.concatenate(agent_pos + entity_gt_pos)

    def compute_macro_allocation(self, world):
        cost = np.zeros((len(world.agents), len(world.landmarks)))
        for agent_id, agent in enumerate(world.agents):
            for landmark_id, entity in enumerate(world.landmarks):  # world.entities:
                rel_dis = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
                cost[agent_id, landmark_id] = rel_dis
            
        row_ind, col_ind = linear_sum_assignment(cost)
        
        dists = 0
        for a in world.agents:
            goal_id = col_ind[a.id]
            target_goal = world.landmarks[goal_id]
            dists += np.sqrt(np.sum(np.square(a.state.p_pos - target_goal.state.p_pos)))
        
        return col_ind, dists

    def info(self, world):
        info = {}
        cover = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            if min(dists) <= world.agents[0].size + l.size:
                cover += 1
        info['success_rate'] = cover / world.num_landmarks      

        return info


