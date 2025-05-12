import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional
from rl_autoschedular import config as cfg
from rl_autoschedular.state import LoopNode
from utils.log import print_alert, print_info, print_success, print_error

def initialization_function_xavier(x):
    return nn.init.xavier_uniform_(x)

class HiearchyModel_old(nn.Module):
    """Hierarchical reinforcement learning model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(HiearchyModel_old, self).__init__()

        L = cfg.max_num_loops
        D = cfg.max_num_load_store_dim
        SD = cfg.max_num_stores_loads
        print(1 + L + L * D * SD + L * D + 5)
        self.input_dim = 1 + L + L * D * SD + L * D + 5 + 7 + L * 4 * cfg.truncate + 7 + 4
        print('input dim:',self.input_dim)
        self.num_loops = L
        self.num_transformations = cfg.num_transformations
        self.num_tiles = cfg.num_tile_sizes

        self.action_mask_size = self.num_transformations + self.num_loops + self.num_loops + self.num_loops + 3 * self.num_loops - 6

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.value_network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.transformation_selection = nn.Linear(512, self.num_transformations)  # +1 for the stop operation
        self.interchange_fc = nn.Linear(512, (3 * self.num_loops - 6))
        self.tiling_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no tiling
        self.parall_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no parallelizattion
        self.fusion_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no fusion

    def sample(self, obs: torch.Tensor, actions: Optional[list[tuple[str, list[int]]]] = None):
        """Sample an action from the model.

        Args:
            obs (torch.Tensor): The input tensor.
            actions (Optional[list[tuple[str, list[int]]]]): list of actions forced for the model to return. Defaults to None.

        Returns:
            list[tuple[str, list[int]]]: list of actions.
            torch.Tensor: action log probabilities.
            torch.Tensor: action values.
            torch.Tensor: resulting entropy.
        """

        *leading_dims, _ = obs.shape

        # Spint `obs` into the input `x` and the `action_mask`
        x = obs[..., :-(self.action_mask_size)]
        action_mask = obs[..., -(self.action_mask_size):].bool()


        # decompose action mask:
        L = self.num_loops

        TP_BEGIN = self.num_transformations
        T_BEGIN = TP_BEGIN + L
        TF_BEGIN = T_BEGIN + L
        I_BEGIN_2C = TF_BEGIN + L
        # I_BEGIN_3C = I_BEGIN_2C + (L - 1)
        # I_BEGIN_4C = I_BEGIN_3C + (L - 2)

        # Define the mask of each transformation
        transform_mask = action_mask[..., :self.num_transformations]
        TP_mask = action_mask[..., TP_BEGIN:T_BEGIN]
        T_mask = action_mask[..., T_BEGIN:TF_BEGIN]
        TF_mask = action_mask[..., TF_BEGIN:I_BEGIN_2C]
        I_mask = action_mask[..., I_BEGIN_2C:]

        # Model inference:
        x1 = self.backbone(x)
        transformation_logits = self.transformation_selection(x1)
        interchange_logits = self.interchange_fc(x1)
        tiling_logits = self.tiling_fc(x1)
        parall_logits = self.parall_fc(x1)
        fusion_logits = self.fusion_fc(x1)

        values = self.value_network(x)

        tiling_logits = tiling_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)
        parall_logits = parall_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)
        fusion_logits = fusion_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)

        # print(parall_logits.shape, tiling_logits.shape, interchange_logits.shape)

        # Apply the mask on the transformations:
        

        transformation_logits = torch.where(transform_mask, transformation_logits, -float('inf'))
        interchange_logits = torch.where(I_mask, interchange_logits, -float('inf'))

        # Get the actions indices:
        transformation_dist = Categorical(logits=transformation_logits)
        interchange_dist = Categorical(logits=interchange_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        parall_dist = Categorical(logits=parall_logits)
        fusion_dist = Categorical(logits=fusion_logits)

        if actions is None:
            transformation_index = transformation_dist.sample()
            interchange_index = interchange_dist.sample()
            fusion_index = fusion_dist.sample()
            tiling_index = tiling_dist.sample()
            parall_index = parall_dist.sample()

        else:

            transformation_index = torch.zeros((len(actions),), dtype=torch.int64)
            parall_index = torch.zeros((len(actions), L), dtype=torch.int64)
            fusion_index = torch.zeros((len(actions), L), dtype=torch.int64)
            tiling_index = torch.zeros((len(actions), L), dtype=torch.int64)
            interchange_index = torch.zeros((len(actions),), dtype=torch.int64)

            for i, action in enumerate(actions):
                action_name, parameters = action
                if action_name == 'no_transformation':
                    transformation_index[i] = 0
                elif action_name == 'parallelization':
                    transformation_index[i] = 1
                    parall_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                elif action_name == 'tiling':
                    transformation_index[i] = 2
                    tiling_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                elif action_name == 'interchange':
                    transformation_index[i] = 3
                    interchange_index[i] = parameters
                elif action_name == 'vectorization':
                    transformation_index[i] = 4
                elif action_name == 'img2col':
                    transformation_index[i] = 5
                elif action_name == 'fusion':
                    fusion_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                    transformation_index[i] = 6

        # Get the action prob and log_prob
        transformation_log_p = F.log_softmax(transformation_logits, dim=-1).gather(-1, transformation_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        interchange_log_p = F.log_softmax(interchange_logits, dim=-1).gather(-1, interchange_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        tiling_log_p = F.log_softmax(tiling_logits, dim=-1).gather(-1, tiling_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        parall_log_p = F.log_softmax(parall_logits, dim=-1).gather(-1, parall_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        fusion_log_p = F.log_softmax(fusion_logits, dim=-1).gather(-1, fusion_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        
        tiling_log_p = torch.where(T_mask, tiling_log_p, 0).sum(-1, keepdim=True)
        parall_log_p = torch.where(TP_mask, parall_log_p, 0).sum(-1, keepdim=True)
        fusion_log_p = torch.where(TF_mask, fusion_log_p, 0).sum(-1, keepdim=True)

        actions = []
        for i in range(transformation_index.shape[0]):
            if transformation_index[i] == 0:
                actions.append(['no_transformation', None])

            elif transformation_index[i] == 1:
                params = []
                for j in range(parall_index[i].shape[0]):
                    if TP_mask[i, j]:
                        params.append(parall_index[i, j].item())
                actions.append(['parallelization', params])

            elif transformation_index[i] == 2:
                params = []
                for j in range(tiling_index[i].shape[0]):
                    if T_mask[i, j]:
                        params.append(tiling_index[i, j].item())
                actions.append(['tiling', params])

            elif transformation_index[i] == 3:
                actions.append(['interchange', interchange_index[i].item()])

            elif transformation_index[i] == 4:
                actions.append(['vectorization', None])

            elif transformation_index[i] == 5:
                actions.append(['img2col', None])
            
            elif transformation_index[i] == 6:
                params = []
                for j in range(fusion_index[i].shape[0]):
                    if TF_mask[i, j]:
                        params.append(fusion_index[i, j].item())
                actions.append(['fusion',params])

        transformation_log_p, interchange_log_p, tiling_log_p, parall_log_p, fusion_log_p = transformation_log_p.reshape(-1), interchange_log_p.reshape(-1), tiling_log_p.reshape(-1), parall_log_p.reshape(-1), fusion_log_p.reshape(-1)

        is_no_action = (transformation_index == 0)
        is_parall = (transformation_index == 1)
        is_tiling = (transformation_index == 2)
        is_interchange = (transformation_index == 3)
        is_fusion = (transformation_index == 6)

        action_log_p = torch.zeros_like(transformation_index, dtype=torch.float32)
        action_log_p[is_interchange] = interchange_log_p[is_interchange] + transformation_log_p[is_interchange]
        action_log_p[is_tiling] = tiling_log_p[is_tiling] + transformation_log_p[is_tiling]
        action_log_p[is_parall] = parall_log_p[is_parall] + transformation_log_p[is_parall]
        action_log_p[is_no_action] = transformation_log_p[is_no_action]
        action_log_p[is_fusion] = fusion_log_p[is_fusion] + transformation_log_p[is_fusion]

        entropy = transformation_dist.entropy().mean() + interchange_dist.entropy().mean() + tiling_dist.entropy().mean() + parall_dist.entropy().mean() + fusion_dist.entropy().mean()

        return actions, action_log_p, values, entropy
        # return action_log_p, entropy, values, sub_entropies

class HiearchyModel(nn.Module):
    """Hierarchical reinforcement learning model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(HiearchyModel, self).__init__()

        L = cfg.max_num_loops
        D = cfg.max_num_load_store_dim
        SD = cfg.max_num_stores_loads
        self.input_dim = 1 + L + L * D * SD + L * D + 5 + L * 3 * cfg.truncate + 6 # TODO: rechange it once the observation vector is finalized
        
        self.comp_embed_layer_sizes=[600, 350, 200, 411] # 411 = 1 + L + L * D * SD + L * D + 5 + 6
        self.drops=[0.225, 0.225, 0.225, 0.225]        
        self.num_loops = L
        self.num_transformations = cfg.num_transformations
        self.num_tiles = cfg.num_tile_sizes
        
        embedding_size = self.comp_embed_layer_sizes[-1]
        
        concat_layer_sizes = [
            embedding_size * 2  # i changed it to *2 only because we dont have the loop_tensor_vector
        ] + self.comp_embed_layer_sizes[-2:]
        
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        
        self.action_mask_size = self.num_transformations + self.num_loops + self.num_loops + 3 * self.num_loops - 6
        
        self.no_comps_tensor = nn.Parameter(torch.randn(1, embedding_size) * 0.01)
        self.no_nodes_tensor = nn.Parameter(torch.randn(1, embedding_size) * 0.01)
        
        
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            self.concat_dropouts.append(nn.Dropout(self.drops[i]))
            
        self.ELU = nn.ELU()
        
        self.comps_lstm = nn.LSTM(
            self.comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        
        # LSTM to encode child loop levels
        self.nodes_lstm = nn.LSTM(
            self.comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        
        self.roots_lstm = nn.LSTM(
            self.comp_embed_layer_sizes[-1], 376, batch_first=True # 376 + 140 (action history) = 516
        )
        
        

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.value_network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.transformation_selection = nn.Linear(512, self.num_transformations)  # +1 for the stop operation
        self.interchange_fc = nn.Linear(512, (3 * self.num_loops - 6))
        self.tiling_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no tiling
        self.parall_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no parallelizattion
        self.fusion_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no fusion
    
    
    def get_hidden_state(self, node):
        if node is not None and node.children != []:
            nodes_list = []

            for n in node.children:
                # Recusrive call to embed all the children of the loop first if they exist
                nodes_list.append(self.get_hidden_state(n))
        
            # Pass the embedding of all the child loops through the nodes LSTM
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        
        else: # If there are no child loops contained within this level
            # The nodes embedding is a random vector (no_nodes_tensor) that represents that there are no nodes underneath this level
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                1, -1, -1
            )

        if node is not None and node.vector is not None:
            # If there are computations contained in this loop, pass them through the computations LSTM
            
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(
                torch.unsqueeze(torch.unsqueeze(torch.tensor(node.vector,dtype=torch.float32),dim=0),dim=0)
            )
            # comps_h_n = comps_h_n.permute(1, 0, 2)
        else: # If there are no child computations contained within this level
            # The computations embedding is a random vector (no_comps_tensor) that represents that there are no computations underneath this level
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                1, # i changed it to 1 for now
                -1, 
                -1
            )
            
        # Concatinate the loop vector, computations embedding and nodes (child loops) embedding
        x = torch.cat((nodes_h_n, comps_h_n), 2)
        # Pass the concatinated vector through a feed forward neural network
        
        
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))

        return x

    def sample(self, obs: tuple[LoopNode, LoopNode], action_mask: np.array, action_history:np.array, actions: Optional[list[tuple[str, list[int]]]] = None):
        """Sample an action from the model.

        Args:
            obs (tuple[LoopNode, LoopNode]): the input representing the current and previous loops.
            actions (Optional[list[tuple[str, list[int]]]]): list of actions forced for the model to return. Defaults to None.

        Returns:
            list[tuple[str, list[int]]]: list of actions.
            torch.Tensor: action log probabilities.
            torch.Tensor: action values.
            torch.Tensor: resulting entropy.
        """
        
        current_tree, previous_tree = obs
        
        current_obs = self.get_hidden_state(current_tree)
        previous_obs = self.get_hidden_state(previous_tree)
        
        roots_tensor = torch.cat([current_obs,previous_obs], 1)
        
        lstm_out, (roots_h_n, roots_c_n) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        action_history = torch.tensor(action_history, dtype=torch.float32).unsqueeze(0)
        

        x = roots_h_n[0]
       
        x = torch.cat([x,action_history],1)
        
        
        
        *leading_dims, _ = x.shape

        # Spint `obs` into the input `x` and the `action_mask`
        # x = obs[..., :-(self.action_mask_size)]
        # action_mask = obs[..., -(self.action_mask_size):].bool()

        # print(action_mask)

        # decompose action mask:
        L = self.num_loops

        TP_BEGIN = self.num_transformations
        T_BEGIN = TP_BEGIN + L
        TF_BEGIN = T_BEGIN + L
        I_BEGIN_2C = TF_BEGIN + L
        # I_BEGIN_3C = I_BEGIN_2C + (L - 1)
        # I_BEGIN_4C = I_BEGIN_3C + (L - 2)

        # TODO: make action_mask an argument
        action_mask = torch.tensor(np.expand_dims(action_mask, axis=0))

        # Define the mask of each transformation
        transform_mask = action_mask[..., :self.num_transformations]
        TP_mask = action_mask[..., TP_BEGIN:T_BEGIN]
        TF_mask = action_mask[..., T_BEGIN:TF_BEGIN]
        T_mask = action_mask[..., TF_BEGIN:I_BEGIN_2C]
        I_mask = action_mask[..., I_BEGIN_2C:]

        # Model inference:
        
        # output, (h_n,c_n) = self.lstm(input)
        # embedding shape = (num layer, batch, hidden) if batch_first = True
        # _, ( embedding,  _) = self.lstm(input) # output shape = (batch, seq_lenght, hidden) input shape = (batch, seq, input size)

        x1 = self.backbone(x)
        transformation_logits = self.transformation_selection(x1)
        interchange_logits = self.interchange_fc(x1)
        tiling_logits = self.tiling_fc(x1)
        parall_logits = self.parall_fc(x1)
        fusion_logits = self.fusion_fc(x1)

        values = self.value_network(x)

        tiling_logits = tiling_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)
        parall_logits = parall_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)
        fusion_logits = fusion_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)

        # print(parall_logits.shape, tiling_logits.shape, interchange_logits.shape)

        # Apply the mask on the transformations:
        transformation_logits = torch.where(transform_mask, transformation_logits, -float('inf'))
        interchange_logits = torch.where(I_mask, interchange_logits, -float('inf'))

        # Get the actions indices:
        transformation_dist = Categorical(logits=transformation_logits)
        interchange_dist = Categorical(logits=interchange_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        parall_dist = Categorical(logits=parall_logits)
        fusion_dist = Categorical(logits=fusion_logits)

        if actions is None:
            transformation_index = transformation_dist.sample()
            interchange_index = interchange_dist.sample()
            tiling_index = tiling_dist.sample()
            parall_index = parall_dist.sample()
            fusion_index = fusion_dist.sample()

        else:

            transformation_index = torch.zeros((len(actions),), dtype=torch.int64)
            parall_index = torch.zeros((len(actions), L), dtype=torch.int64)
            fusion_index = torch.zeros((len(actions), L), dtype=torch.int64)
            tiling_index = torch.zeros((len(actions), L), dtype=torch.int64)
            interchange_index = torch.zeros((len(actions),), dtype=torch.int64)

            for i, action in enumerate(actions):
                action_name, parameters = action
                if action_name == 'no_transformation':
                    transformation_index[i] = 0
                elif action_name == 'parallelization':
                    transformation_index[i] = 1
                    parall_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                elif action_name == 'tiling':
                    transformation_index[i] = 2
                    tiling_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                elif action_name == 'interchange':
                    transformation_index[i] = 3
                    interchange_index[i] = parameters
                elif action_name == 'vectorization':
                    transformation_index[i] = 4
                elif action_name == 'img2col':
                    transformation_index[i] = 5
                elif action_name == "fusion":
                    fusion_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                    transformation_index[i] = 6


        # Get the action prob and log_prob
        transformation_log_p = F.log_softmax(transformation_logits, dim=-1).gather(-1, transformation_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        interchange_log_p = F.log_softmax(interchange_logits, dim=-1).gather(-1, interchange_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        tiling_log_p = F.log_softmax(tiling_logits, dim=-1).gather(-1, tiling_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        parall_log_p = F.log_softmax(parall_logits, dim=-1).gather(-1, parall_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        fusion_log_p = F.log_softmax(fusion_logits, dim=-1).gather(-1, fusion_index.unsqueeze(-1)).reshape(*leading_dims, -1)

        tiling_log_p = torch.where(T_mask, tiling_log_p, 0).sum(-1, keepdim=True)
        parall_log_p = torch.where(TP_mask, parall_log_p, 0).sum(-1, keepdim=True)
        fusion_log_p = torch.where(TF_mask, fusion_log_p, 0).sum(-1, keepdim=True)

        actions = []
        for i in range(transformation_index.shape[0]):
            if transformation_index[i] == 0:
                actions.append(['no_transformation', None])

            elif transformation_index[i] == 1:
                params = []
                for j in range(parall_index[i].shape[0]):
                    if TP_mask[i, j]:
                        params.append(parall_index[i, j].item())
                actions.append(['parallelization', params])

            elif transformation_index[i] == 2:
                params = []
                for j in range(tiling_index[i].shape[0]):
                    if T_mask[i, j]:
                        params.append(tiling_index[i, j].item())
                actions.append(['tiling', params])

            elif transformation_index[i] == 3:
                actions.append(['interchange', interchange_index[i].item()])

            elif transformation_index[i] == 4:
                actions.append(['vectorization', None])

            elif transformation_index[i] == 5:
                actions.append(['img2col', None])

            elif transformation_index[i] == 6:
                params = []
                for j in range(fusion_index[i].shape[0]):
                    if TF_mask[i, j]:
                        params.append(fusion_index[i, j].item())
                actions.append(['fusion', params])

        transformation_log_p, interchange_log_p, tiling_log_p, parall_log_p, fusion_log_p = transformation_log_p.reshape(-1), interchange_log_p.reshape(-1), tiling_log_p.reshape(-1), parall_log_p.reshape(-1), fusion_log_p.reshape(-1)

        is_no_action = (transformation_index == 0)
        is_parall = (transformation_index == 1)
        is_tiling = (transformation_index == 2)
        is_interchange = (transformation_index == 3)
        is_fusion = (transformation_index == 6)

        action_log_p = torch.zeros_like(transformation_index, dtype=torch.float32)
        action_log_p[is_interchange] = interchange_log_p[is_interchange] + transformation_log_p[is_interchange]
        action_log_p[is_tiling] = tiling_log_p[is_tiling] + transformation_log_p[is_tiling]
        action_log_p[is_parall] = parall_log_p[is_parall] + transformation_log_p[is_parall]
        action_log_p[is_fusion] = fusion_log_p[is_fusion] + transformation_log_p[is_fusion]
        action_log_p[is_no_action] = transformation_log_p[is_no_action]

        entropy = transformation_dist.entropy().mean() + interchange_dist.entropy().mean() + tiling_dist.entropy().mean() + parall_dist.entropy().mean() + fusion_dist.entropy().mean()

        return actions, action_log_p, values, entropy
        # return action_log_p, entropy, values, sub_entropies
