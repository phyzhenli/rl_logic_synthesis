import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GINConv, SumPooling, AvgPooling, MaxPooling
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor, create_mlp
from stable_baselines3.common.policies import BasePolicy, register_policy
from sb3_contrib.ars.policies import ARSPolicy

def MLP(num_layers, in_dim, hidden_dim, out_dim):
    modules = create_mlp(in_dim, out_dim, [hidden_dim] * (num_layers-1))
    return nn.Sequential(*modules)

class GCN(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcs = torch.nn.ModuleList()
        for i in range(num_layers):
            insz = in_dim if i == 0 else hidden_dim
            outsz = hidden_dim if i < num_layers - 1 else out_dim
            self.gcs.append(GraphConv(insz, outsz))

    def forward(self, g):
        h = g.ndata['feat']
        w = g.edata['w'] if 'w' in g.edge_attr_schemes() else None
        for i in range(self.num_layers - 1):
            h = F.relu(self.gcs[i](g, h, edge_weight = w))
        h = self.gcs[-1](g, h, edge_weight = w)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')


class GIN(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.fcs  = torch.nn.ModuleList()
        self.gins = torch.nn.ModuleList()
        for i in range(num_layers):
            insz = in_dim if i == 0 else hidden_dim
            self.fcs.append(nn.Linear(insz, out_dim))
            if i < num_layers - 1:
                mlp = MLP(2, insz, hidden_dim, hidden_dim)
                self.gins.append(GINConv(mlp, 'sum'))
        self.pool = SumPooling()

    def forward(self, g):
        h = g.ndata['feat']
        w = g.edata['w'] if 'w' in g.edge_attr_schemes() else None
        score_over_layer = 0
        for i in range(self.num_layers):
            pooled_h = self.pool(g, h)
            score_over_layer += self.fcs[i](pooled_h)
            if i < self.num_layers - 1:
                h = self.gins[i](g, h, w)
                h = F.relu(h)
        return F.normalize(score_over_layer)

class GNone(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        assert out_dim == 0
        self.out_dim = out_dim

    def forward(self, g):
        return torch.tensor([]).to(g.device)

def GNN(type):
    if type == 'GCN': return GCN
    if type == 'GIN': return GIN
    return GNone

class CombinedGraphExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, state_out = 24,
                 gnn_type = 'GCN', gnn_out = 8, gnn_layers = 2, gnn_hidden = 24, dummy = False):
        self.dummy = dummy
        features_dim = state_out + gnn_out if not dummy else observation_space['state'].shape[0]
        super().__init__(observation_space, features_dim = features_dim)
        self.gnn_type = gnn_type
        self.state_extractor = nn.Sequential(nn.Linear(observation_space['state'].shape[0], state_out), nn.ReLU())
        self.graph_extractor = GNN(gnn_type)(gnn_layers, observation_space['ndfeat'].shape[1], gnn_hidden, gnn_out)

    def forward(self, observations) -> torch.Tensor:
        if self.dummy: return observations['state']
        batch_size = observations['gdim'].shape[0]
        graphs = []
        for i in range(batch_size):
            edges, nodes = (int(a) for a in observations['gdim'][i])
            esrc  = torch.narrow(observations['esrc'][i], 0, 0, edges)
            edst  = torch.narrow(observations['edst'][i], 0, 0, edges)
            g = dgl.graph((esrc, edst), idtype=torch.int32, num_nodes = nodes)
            g.ndata['feat'] = torch.narrow(observations['ndfeat'][i], 0, 0, nodes)
            if 'ewgt' in observations:
                g.edata['w'] = torch.narrow(observations['ewgt'], 0, 0, edges)
            if self.gnn_type == 'GCN':
                g = dgl.add_self_loop(g)
            graphs.append(g)

        result_list = [self.state_extractor(observations['state']), self.graph_extractor(dgl.batch(graphs))]
        #print(result_list)
        return torch.cat(result_list, dim=1)


class MultiInputARSPolicy(ARSPolicy):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Space,
        net_arch = [64,64], activation_fn = nn.ReLU, squash_output = True,
        features_extractor_class = CombinedExtractor,
        features_extractor_kwargs = None
    ):
        BasePolicy.__init__(self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            squash_output=isinstance(action_space, gym.spaces.Box) and squash_output,
        )
        self.net_arch = net_arch
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.activation_fn = activation_fn

        if isinstance(action_space, gym.spaces.Box):
            action_dim = get_action_dim(action_space)
            actor_net = create_mlp(self.features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        elif isinstance(action_space, gym.spaces.Discrete):
            actor_net = create_mlp(self.features_dim, action_space.n, net_arch, activation_fn)
        else:
            raise NotImplementedError(f"Error: ARS policy not implemented for action space of type {type(action_space)}.")

        self.action_net = nn.Sequential(*actor_net)


register_policy("MultiInputPolicy", MultiInputARSPolicy)