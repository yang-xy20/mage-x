import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
from scipy.optimize import linear_sum_assignment

class Scenario(BaseScenario):
    def make_world(self, args, now_agent_num=None):
        world = World()
        # set any world properties first
        world.name = 'ball'
        world.dim_c = 2
        if now_agent_num==None:
            num_people = args.num_agents
            num_boxes = args.num_landmarks
            num_landmarks = args.num_landmarks
        else:
            num_people = now_agent_num
            num_boxes = now_agent_num
            num_landmarks = now_agent_num
        self.num_boxes = num_boxes
        self.num_people = num_people
        self.num_agents = num_boxes + num_people # deactivate "good" agent
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.id = i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_people else False  # people.adversary = True     box.adversary = False
            agent.size = 0.1 if agent.adversary else 0.15
            # agent.accel = 3.0 if agent.adversary else 5
            # agent.max_speed = 0.5 if agent.adversary else 0.5
            agent.action_callback = None if i < num_people else self.box_policy  # box有action_callback 即不做动作

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15
            landmark.cover = 0
            # landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        world.pred_box_id = np.zeros((self.num_people))
        world.pred_land_id = np.zeros((self.num_people))
        world.use_exe_gnn = args.use_exe_gnn
        return world

    def box_policy(self, agent, world):
        # action of the agent
        chosen_action = np.array([0,0], dtype=np.float32)
        # chosen_action_c = np.array([0,0], dtype=np.float32)
        return chosen_action

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0, 0, 0])
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-4.0, +4.0, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.prev_agent_state = []            
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-4.0, +4.0, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if agent.id < self.num_people:
                self.prev_agent_state.append(agent.state.p_pos)

        self.ab_id, self.bl_id, self.gt_dists = self.compute_macro_allocation(world)

    def landmark_cover_state(self, world):
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) <= world.agents[0].size + world.landmarks[0].size:
                l.cover = 1
            else:
                l.cover = 0

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    def reset_radius(self,sample_radius):
        sample_radius = max(sample_radius,1.5)
        self.sample_radius = sample_radius

    def get_state(self, world):
        pass

    def info(self, world):
        num = 0
        success = False
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents if a.adversary == False]
            if min(dists) <= world.agents[-1].size + world.landmarks[0].size:
                num = num + 1
        # success
        # if num==len(world.landmarks):
        #     success = True
        info_list = {'success_rate': num/len(world.landmarks)}
        return info_list

    
    def reward(self, agent, world, mode):
    # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    
        rew = 0
        cover = 0
        if mode == 'exe':
            box_id = world.pred_box_id[agent.id]
            land_id = world.pred_land_id[agent.id]
            target_box = world.agents[int(box_id)+self.num_people]    
            target_goal = world.landmarks[int(land_id)]
            dists_ab = np.sqrt(np.sum(np.square(agent.state.p_pos - target_box.state.p_pos)))
            dists_bl = np.sqrt(np.sum(np.square(target_box.state.p_pos - target_goal.state.p_pos)))
            rew -= (dists_ab + dists_bl)
            # if dists_ab <= world.agents[0].size + world.landmarks[0].size:
            #     rew += 0.5
            if dists_bl <= 2*world.landmarks[0].size:
                rew += 1

        elif mode == 'ctl':
            dists = 0
            if not np.any(world.pred_box_id):
                rew = 0
            else:
                for agent_id in range(self.num_people):
                    box_id = world.pred_box_id[agent_id]
                    land_id = world.pred_land_id[agent_id]
                    target_box = world.agents[int(box_id)+self.num_people]    
                    target_goal = world.landmarks[int(land_id)]
                    dists += np.sqrt(np.sum(np.square(self.prev_agent_state[agent_id] - target_box.state.p_pos)))+\
                    np.sqrt(np.sum(np.square(target_box.state.p_pos-target_goal.state.p_pos)))
                rew = -(dists - self.gt_dists)
                self.prev_agent_state = []
                for a in world.agents:
                    self.prev_agent_state.append(a.state.p_pos)
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
        other_pos = []
        other_gt_pos = []
        box_id = world.pred_box_id[agent.id]
        goal_id = world.pred_land_id[agent.id]
        target_gt_box = world.agents[int(box_id)+self.num_people].state.p_pos
        target_box = world.agents[int(box_id)+self.num_people].state.p_pos - agent.state.p_pos
        for land_id, entity in enumerate(world.landmarks):  # world.entities:
            if int(goal_id) == int(land_id):
                target_gt_goal = entity.state.p_pos
                target_goal = entity.state.p_pos - agent.state.p_pos
                for agent_id, other in enumerate(world.agents):
                    agent_pos.append(other.state.p_pos)
                    if other is agent:
                        continue
                    if agent_id < self.num_people:
                        other_pos.append(other.state.p_pos - agent.state.p_pos)
                        other_gt_pos.append(other.state.p_pos)
            entity_gt_pos.append(entity.state.p_pos)
            
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents        
        id_vector = np.zeros(self.num_people)
        id_vector[agent.id] = 1
        if mode == 'exe':
            if world.use_exe_gnn:
                info = {}
                info['agent_state'] = np.concatenate([agent.state.p_vel]+[agent.state.p_pos])
                info['target_goal'] = np.concatenate([target_gt_box]+[target_gt_goal])
                info['other_pos'] = np.stack(other_gt_pos)
                # info['agent_id'] = np.stack(id_vector)
                return info
            else:
                return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [target_box] + [target_goal] + other_pos)
            #entity_pos + other_pos + [id_vector] + comm)
        elif mode == 'ctl':
                return np.concatenate(agent_pos + entity_gt_pos)

    def compute_macro_allocation(self, world):
        cost_agent_box = np.zeros((len(world.agents), len(world.landmarks)))
        cost_box_land = np.zeros((len(world.agents), len(world.landmarks)))
        for box_id in range(self.num_boxes):
            box = world.agents[box_id+self.num_boxes]
            for landmark_id, entity in enumerate(world.landmarks):  # world.entities:
                rel_dis = np.sqrt(np.sum(np.square(entity.state.p_pos - box.state.p_pos)))
                cost_box_land[box_id, landmark_id] = rel_dis
            for people_id in range(self.num_people):
                people = world.agents[people_id]
                rel_dis = np.sqrt(np.sum(np.square(people.state.p_pos - box.state.p_pos)))
                cost_agent_box[people_id, box_id] = rel_dis
                
        bl_row_ind, bl_col_ind = linear_sum_assignment(cost_box_land)
        ab_row_ind, ab_col_ind = linear_sum_assignment(cost_agent_box)
        
        dists = 0
        for box_id in range(self.num_boxes):
            box = world.agents[box_id+self.num_boxes]
            agent_id = ab_row_ind[box_id]
            land_id = bl_col_ind[box_id]
            target_agent = world.agents[agent_id]
            target_land = world.landmarks[land_id]
            dists += np.sqrt(np.sum(np.square(box.state.p_pos - target_agent.state.p_pos)))+\
            np.sqrt(np.sum(np.square(box.state.p_pos - target_land.state.p_pos)))
        return ab_row_ind, bl_col_ind, dists