import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from datetime import date
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""
def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# why retain graph? Do not auto free memory for one loss when computing multiple loss
# https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
def update_params(optim, loss):
    optim.zero_grad()
    loss.backward(retain_graph=True)
    optim.step()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)




class ReplayBuffer:
    def __init__(self, memory_size, eta=0.999, cmin=2000):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0
        self.eta = eta  
        self.cmin = cmin  
        self.first_update = True  

    def get_size(self):
        return len(self.storage)

    def add(self, obs, action, reward, obs_, done, transition_info):
        data = (obs, action, reward, obs_, done, transition_info)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size, total_updates=6250):
        n = self.get_size()  
        print(f"Sampling Step: update_num={self.next_idx}, buffer_size={n}, total_updates={total_updates}")


        
        if n <= self.cmin:
            c_k = n  
            print(f"Uniform Sampling: Using full buffer c_k = {c_k}")
        else:
            update_num = self.next_idx  
            c_k_before = int(n * (self.eta ** ((update_num - self.cmin) * 1000 / total_updates)))
            print(f"Before decay: c_k = {c_k_before}, cmin = {self.cmin}")

            c_k = max(c_k_before, self.cmin)
            print(f"Final c_k after clipping: {c_k}")

            #update_num = self.next_idx
            #c_k = max(int(n * (self.eta ** (update_num * 1000 / total_updates))), self.cmin)
            #print(f"Using ERE with c_k = {float(c_k):.4f}")

        
        indices_range = range(max(n - c_k, 0), n)
        indices = random.sample(indices_range, min(batch_size, len(indices_range)))

        return self._encode_sample(indices)

    def _encode_sample(self, indices):
        obses, actions, rewards, obses_, dones, transition_infos = [], [], [], [], [], []
        for i in indices:
            data = self.storage[i]
            obs, action, reward, obs_, done, transition_info = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
            transition_infos.append(transition_info)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones), np.array(transition_infos)


    # Save Buffer
    def save_buffer(self, path, name):
        path = os.path.join(path, name)
        self.storage = np.array(self.storage, dtype=object)
        np.save(path, self.storage)

    # Load Buffer
    def load_buffer(self, path):
        self.storage = np.load(path, allow_pickle=True).tolist()

    def merge_buffers(self, args):
        buffer1 = np.load(args.buffer_path_1, allow_pickle=True).tolist()
        buffer2 = np.load(args.buffer_path_2, allow_pickle=True).tolist()
        if args.buffer_path_3 is not None:
            buffer3 = np.load(args.buffer_path_3, allow_pickle=True).tolist()
            self.storage = buffer1 + buffer2 + buffer3
        else:
            self.storage = buffer1 + buffer2
        self.next_idx = len(self.storage)



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='actor', chkpt_dir='tmp/sac',load_file=None):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.load_file = load_file
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac'+ str(date.today()))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.name = name
        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[0], n_hidden_units[1]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[1], action_dim)
        ).apply(init_weights)

    def forward(self, s):
        actions_logits = self.actor_mlp(s)
        return F.softmax(actions_logits, dim=-1)

    def greedy_act(self, s):  # no softmax more efficient
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        greedy_actions = torch.argmax(actions_logits, dim=-1, keepdim=True)
        return greedy_actions.item()

    def sample_act(self, s):
        s = torch.from_numpy(s).float().to(device)
        actions_logits = self.actor_mlp(s)
        actions_probs = F.softmax(actions_logits, dim=-1)
        actions_distribution = Categorical(actions_probs)
        action = actions_distribution.sample()

        arg_max_action = torch.argmax(actions_probs)

        return action.item(),arg_max_action.item()

    def save_checkpoint(self,block):
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir,str(block)+'_actor.pt'))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.load_file+'_actor.pt',map_location=device))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='critic', chkpt_dir='tmp/sac',load_file = None):
        super(Critic, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac'+ str(date.today()))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.load_file = load_file

        # self.qnet1 = DuelQNet(state_dim, action_dim, n_hidden_units)
        # self.qnet2 = DuelQNet(state_dim, action_dim, n_hidden_units)
        self.qnet1 = QNet(state_dim, action_dim, n_hidden_units)
        self.qnet2 = QNet(state_dim, action_dim, n_hidden_units)

    def forward(self, s):  # S: N x F(state_dim) -> Q: N x A(action_dim) Q(s,a)
        q1 = self.qnet1(s)
        q2 = self.qnet2(s)
        return q1, q2

    def save_checkpoint(self,block):
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir,str(block)+'_critic.pt'))

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.load_file+'_critic.pt',map_location=device))

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='QNet', chkpt_dir='tmp/sac'):
        super(QNet, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[0], n_hidden_units[1]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[1], action_dim)
        ).apply(init_weights)

    def forward(self, s):
        q_value = self.shared_mlp(s)
    
        return q_value

class DuelQNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='DuelQNet', chkpt_dir='tmp/sac'):
        super(DuelQNet, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir

        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[0], n_hidden_units[1]),
            nn.ReLU()
        ).apply(init_weights)

        # self.q_head = nn.Linear(n_hidden_units, action_dim)

        self.action_head = nn.Linear(n_hidden_units[1], action_dim).apply(init_weights)
        self.value_head = nn.Linear(n_hidden_units[1], 1).apply(init_weights)

    def forward(self, s):
        s = self.shared_mlp(s)
        a = self.action_head(s)
        v = self.value_head(s)
        return v + a - a.mean(1, keepdim=True)
        # return self.q_head(s)