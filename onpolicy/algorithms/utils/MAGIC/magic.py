import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .gnn_layers import GraphAttention
from onpolicy.algorithms.utils.MAGIC.graph_conv_module import  GraphConvolutionModule
from onpolicy.algorithms.utils.MAGIC.invariant import Invariant

class Topk_Graph(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, num_agents, hidden_size, use_attn, device=torch.device("cuda:0")):
        super(Topk_Graph, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.num_agents = num_agents
        self.hidden_size = hidden_size
        self.use_attn = use_attn
        gat_encoder_out_size = 32
        gat_hid_size = 32
        
        dropout = 0
        negative_slope = 0.2
        if self.use_attn:
            self.attn_n = Invariant(hidden_dim = self.hidden_size, heads = 4, dim_head = 16, mlp_dim = 64)
        else:
            # initialize sub-processors
            self.sub_processor1 = GraphConvolutionModule(self.hidden_size, gat_hid_size)
            self.sub_processor2 = GraphConvolutionModule(gat_hid_size, self.hidden_size)
        
            self.gat_encoder = GraphConvolutionModule(self.hidden_size, gat_encoder_out_size)
                
            # initialize the gat encoder for the Scheduler
        self.obs_encoder = nn.Linear(2, self.hidden_size)#

        self.land_encoder = nn.Linear(6, self.hidden_size)#
        self.all_encoder = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.hidden_size))
        
        # initialize mlp layers for the sub-schedulers
        self.sub_scheduler_mlp1 = nn.Sequential(
            nn.Linear(gat_encoder_out_size*2, gat_encoder_out_size//2),
            nn.ReLU(),
            nn.Linear(gat_encoder_out_size//2, gat_encoder_out_size//2),
            nn.ReLU(),
            nn.Linear(gat_encoder_out_size//2, 2))

    
        self.sub_scheduler_mlp2 = nn.Sequential(
            nn.Linear(gat_encoder_out_size*2, gat_encoder_out_size//2),
            nn.ReLU(),
            nn.Linear(gat_encoder_out_size//2, gat_encoder_out_size//2),
            nn.ReLU(),
            nn.Linear(gat_encoder_out_size//2, 2))

        self.device = device

    def forward(self, obs):
        """
        Forward function of MAGIC (two rounds of communication)

        Arguments:
            x (list): a list for the input of the communication protocol [observations, (previous hidden states, previous cell states)]
            observations (tensor): the observations for all agents [1 (batch_size) * n * obs_size]
            previous hidden/cell states (tensor): the hidden/cell states from the previous time steps [n * hid_size]

        Returns:
            action_out (list): a list of tensors of size [1 (batch_size) * n * num_actions] that represent output policy distributions
            value_head (tensor): estimated values [n * 1]
            next hidden/cell states (tensor): next hidden/cell states [n * hid_size]
        """
        # n: number of agents
        #obs, rnn_state, masks = x
        # encoded_obs: [1 (batch_size) * n * hid_size]
        if self.use_attn:
            inter_message = self.attn(obs)
        else:
            inter_message = self.agent_interaction(obs)
        agent_land = torch.cat((obs['agent_state'], obs['target_goal']), dim = 2)
        land_embedding = self.land_encoder(agent_land)[:,0]
        all_embedding = torch.cat((inter_message, land_embedding), dim = -1)
        message = self.all_encoder(all_embedding)
        return message

    def attn(self, obs):
        if obs['other_pos'].shape[-1] != 2:
            all_state = torch.cat((obs['agent_state'], obs['other_pos']),dim = 1)
        else:
            all_state = torch.cat((obs['agent_state'][:,:,:2], obs['other_pos']),dim = 1)
        encoded_obs = self.obs_encoder(all_state)
        x = self.attn_n(encoded_obs)
        return x

    def agent_interaction(self, obs):
        if obs['other_pos'].shape[-1] != 2:
            all_state = torch.cat((obs['agent_state'], obs['other_pos']),dim = 1)
        else:
            all_state = torch.cat((obs['agent_state'][:,:,:2], obs['other_pos']),dim = 1)
        encoded_obs = self.obs_encoder(all_state)
        #hidden_state, cell_state = extras
        batch_size = encoded_obs.size()[0]
        n = self.num_agents
        e_obs = encoded_obs.reshape(batch_size*n ,-1)

        # comm: [n * hid_size]
        comm = e_obs
       
        # mask communcation from dead agents (only effective in Traffic Junction)
        # sub-scheduler 1
        adj_complete = self.get_complete_graph(batch_size).to(self.device)
        comm = comm.view(batch_size, n, self.hidden_size)
        encoded_state1 = self.gat_encoder(comm, adj_complete)
        adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, encoded_state1, True)
        # sub-processor 1
        comm = F.elu(self.sub_processor1(comm, adj1))
        
        # sub-scheduler 2
        encoded_state2 = encoded_state1
        adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, encoded_state2, True)
        # sub-processor 2
        comm = self.sub_processor2(comm, adj2)
        comm = comm[:,0]
        return comm 

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
  
    def sub_scheduler(self, sub_scheduler_mlp, hidden_state, directed=True):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]
            directed (bool): decide if generate directed graphs

        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]  
        """

        # hidden_state: [n * hid_size]
        n = self.num_agents
        batch_size = hidden_state.size(0)
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, 1, n).view(batch_size, n * n, -1), hidden_state.repeat(1, n, 1)], dim=1).view(batch_size, n, -1, 2 * hid_size)
        # hard_attn_output: [n * n * 2]
        if directed:
            hard_attn_output = F.gumbel_softmax(sub_scheduler_mlp(hard_attn_input), hard=True)
        else:
            hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 3, 1, 1)
        # adj: [n * n]
        adj = hard_attn_output.squeeze() 
        
        return adj
    
    def get_complete_graph(self,batch_size):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.num_agents
        adj = torch.ones(batch_size, n, n)
     
        return adj
        
    @property
    def output_size(self):
        output_size = self.hidden_size
        return output_size
