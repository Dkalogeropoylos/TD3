import os

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
from datetime import date
import random
from collections import deque, namedtuple
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
# Debugging and Adjusting the Sample Method to Resolve the ValueError

import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, eta=0.999,cmin=1500):
        self.storage = deque(maxlen=buffer_size)  # Automatically handles buffer overflow
        self.memory_size = buffer_size
        self.eta = eta  # Decay rate for ERE
        self.cmin = cmin  # Minimum samples for ERE
        self.first_update = True  # Flag for first update
        self.update_num = 0
        self.ere_active = False
        self.ere_update_count = None
        self.adj_total_updates = 0
        
        
    def add(self, obs, action, reward, obs_, done,transition_info):
        vector_action = self.convert_action_to_vector(action)
        #print(f"Vector Action: {vector_action}")
        #print(vector_action)
        
        data = (obs,action,reward, obs_,done,transition_info)
        self.storage.append(data)
    #def add(self, obs, action, reward, obs_, done, transition_info):
            # Convert action to vector and ensure consistency
        #vector_action = self.convert_action_to_vector(action)
        #if isinstance(vector_action, list):
            #vector_action = np.array(vector_action)

        # Standardize transition_info
        #if isinstance(transition_info, list):
            #transition_info = np.array(transition_info, dtype=object)

        # Debugging: Check the shapes of all elements
        #data = (obs, vector_action, reward, obs_, done, transition_info)
        #print(f"Adding Data: {data}, Shapes: {[np.shape(x) if hasattr(x, 'shape') else type(x) for x in data]}")

        # Add data to storage
        #self.storage.append(data)

     
    def convert_action_to_vector(self, action):
        if action is None:
            #print("Action is None; interpreting as -1")
            action = 2  # Convert None to -1 explicitly
        action_map = {
            2: np.array([1, 0, 0], dtype=np.float32),
            0: np.array([0, 1, 0], dtype=np.float32),
            1: np.array([0, 0, 1], dtype=np.float32),
            }
        vector_action = action_map.get(action)
        #print("why")
        #if vector_action is None:
            #print(f"Unexpected action value: {action}")
    
        return vector_action    

    

   
    def _encode_sample(self, idx, demo=False):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            if  demo:
                data = self.expert_storage[i]
            else:
                data = self.storage[i]
            obs, action, reward, obs_, done,transition_info = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones),np.array(transition_info)
    
    
    
    
    
    
    
    
    def sample1(self, batch_size, update_num=None, total_updates=1000):
        """
        Samples a batch using ERE or regular sampling logic.
        """
        n = len(self.storage)  # Current buffer size

        if self.first_update:
            c_k = n  # Use the entire buffer for the first update
            self.first_update = False
            #print("First update: Using the entire buffer for sampling.")
        else:
            if update_num is not None:
                c_k = max(int(n * (self.eta ** (update_num / total_updates))), self.cmin)
                #print(f"ERE sampling: c_k={c_k}, eta={self.eta}, cmin={self.cmin}")
            else:
                c_k = n

        indices_range = range(max(n - c_k, 0), n)
        indices = random.sample(indices_range, min(batch_size, len(indices_range)))

        return self._encode_sample(indices)
    
  
 
    
    def sample(self, batch_size, update_num=None ,total_updates=6250):
        n = self.get_size()
        print(n)
        



        if n <= self.cmin:
            c_k = n
            print(f"Uniform Sampling: Using full buffer c_k = {c_k}")
        else:
            if not self.ere_active:
                self.ere_active = True
                self.ere_update_count = 1
                self.adj_total_updates= 6251 - update_num
                print(f"adjusted total updates: {self.adj_total_updates}")





            #c_k_before = int(n * (self.eta ** (self.ere_update_count  * 1000 / self.adj_total_updates)))
            #self.ere_update_count = 1 + self.ere_update_count
            #print(f"Before decay: c_k = {c_k_before}, cmin = {self.cmin}")
            print(int(n * (self.eta ** (self.ere_update_count  * 1000 / self.adj_total_updates))))
            c_k = max(int(n * (self.eta ** (self.ere_update_count  * 1000 / self.adj_total_updates))), self.cmin)
            self.ere_update_count = 1 + self.ere_update_count
            print(f"Final c_k after clipping: {c_k}")


            #update_num = self.next_idx
            #c_k = max(int(n * (self.eta ** (update_num * 1000 / total_updates))), self.cmin)
            #print(f"Using ERE with c_k = {float(c_k):.4f}")


        indices_range = range(max(n - c_k, 0), n)
        indices = random.sample(indices_range, min(batch_size, len(indices_range)))
        return self._encode_sample(indices)    




    # Get the current size of the buffer
    def get_size(self):
        return len(self.storage)

  



     #Save the buffer to a file
    def save_buffer(self, path, name):
       
        path = os.path.join(path, name)
        
        self.storage = np.array(self.storage, dtype=object)
        np.save(path, self.storage)

        #np.save(path, list(self.storage))
        
        
            
    
    
    
    #def save_buffer(self, path):
        #try:
            #processed_data = []
            #for item in self.storage:
               # obs, action, reward, obs_, done, _ = item  # Ignore transition_info by assigning it to _
               # processed_item = (
                   # np.array(obs, dtype=np.float32),
                   # np.array(action, dtype=np.int64 if isinstance(action, int) else np.float32),
                   # np.float32(reward),
                   # np.array(obs_, dtype=np.float32),
                   # np.bool_(done)
               # )
               # processed_data.append(processed_item)
        
        # Save the processed data to a NumPy file
            #np.save(path, processed_data)
            #print(f"Buffer saved successfully to {path}")
       # except Exception as e:
           # print(f"Error saving buffer: {e}")
      
        
    

    # Load a saved buffer from a file
    def load_buffer(self, path):
        self.storage = deque(np.load(path, allow_pickle=True).tolist(), maxlen=self.memory_size)


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
    def __init__(self, state_dim, action_dim, n_hidden_units, max_action=2.0, name='actor', chkpt_dir='tmp/td3', load_file=None):
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.load_file = load_file
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.name = name
        self.max_action = max_action  #used for clamping 

        self.actor_mlp = nn.Sequential(
            nn.Linear(state_dim, n_hidden_units[0]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[0], n_hidden_units[1]),
            nn.ReLU(),
            nn.Linear(n_hidden_units[1], action_dim)
        )
        self.apply(self.init_weights)

    #def init_weights(self, m):
        #if isinstance(m, nn.Linear):
           # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            #nn.init.zeros_(m.bias)
    def init_weights(self,m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
   
    def forward(self, s):
        actions_logits = self.actor_mlp(s)
        return F.softmax(actions_logits, dim=-1)

    def sample_act(self, state):
       
        state = torch.tensor(state, dtype=torch.float32).to(device)
        actions_logits = self.actor_mlp(state)
        action_probs = F.softmax(actions_logits, dim=-1)
        action_distribution = Categorical(action_probs)
        
        action = action_distribution.sample()
        
        arg_max_action = torch.argmax(action_probs)

        return action.item(), arg_max_action.item()
    #def sample_act1(self, state, for_training=True):
        #state = torch.tensor(state, dtype=torch.float32).to(device)
        #logits = self.actor_mlp(state)

       

        #if for_training:
        
            #noise = torch.randn_like(logits) * 0.1
            #logits += noise

    
        #discrete_action = torch.round(logits).clip(-1, 1)  # Map to [-1, 0, 1]
        #argmax_action = torch.argmax(logits, dim=-1).item()
        #return discrete_action.item(), argmax_action


    def save_checkpoint(self, block=None):
        # Set default filename if block is not provided
        filename = f"{block}_actor.pt" if block is not None else "actor.pt"
    
    # Construct the full path
        full_path = os.path.join(self.checkpoint_dir, filename)

    # Save the model state dictionary
        try:
            torch.save(self.state_dict(), full_path)
            print(f"Model saved successfully at {full_path}")
        except Exception as e:
            print(f"Failed to save model at {full_path}: {str(e)}")


    def load_checkpoint(self, filename=None):
        if filename is None and self.load_file:
            filename = self.load_file
            print("hello bitches")
        if filename:
            self.load_state_dict(torch.load(filename, map_location=device))
            print("bye bitches")



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_units, name='critic', chkpt_dir='tmp/sac', load_file=None):
        super(Critic, self).__init__()
        
        self.name = name
        self.checkpoint_dir = chkpt_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.load_file = load_file
        #print(f"Critc with input dim: {state_dim + action_dim}")

        self.q1_l1 = nn.Linear(state_dim + action_dim, n_hidden_units[0])
        self.q1_l2 = nn.Linear(n_hidden_units[0], n_hidden_units[1])
        self.q1_out = nn.Linear(n_hidden_units[1], 1)
        
        
        self.q2_l1 = nn.Linear(state_dim + action_dim, n_hidden_units[0])
        

        
        self.q2_l2 = nn.Linear(n_hidden_units[0], n_hidden_units[1])
        self.q2_out = nn.Linear(n_hidden_units[1], 1)

    def forward(self, state, action):
    
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        

        
        sa = torch.cat([state, action], dim=1)
       # print(f"Concatenated shape: {sa.shape}")  
        

       
        q1 = F.relu(self.q1_l1(sa))
        
        
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_out(q1)

       
        q2 = F.relu(self.q2_l1(sa))
        

        
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_out(q2)

        return q1, q2
    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_out(q1)
        return q1


    def save_checkpoint(self, block=None):
        filename = f"{block}_critic.pt" if block else "critic.pt"
        torch.save(self.state_dict(), os.path.join(self.checkpoint_dir, filename))

    def load_checkpoint(self):
        if self.load_file:
            self.load_state_dict(torch.load(self.load_file + '_critic.pt', map_location=device))
