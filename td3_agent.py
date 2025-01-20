import torch
import numpy as np
from rl_models.networks_discrete import update_params, Actor, Critic, ReplayBuffer
import torch.nn.functional as F
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3Agent:
    def __init__(self,args = None, config=None, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, buffer_max_size=1000000, tau=0.005,
                 layer1_size=32, layer2_size=32, batch_size=256,
                 chkpt_dir=None,load_file=None,participant_name=None,ID='First'):
        self.args = args
        self.ID = ID
        

        
        if config is not None:
            self.config = config
            self.mode = config['Experiment']['mode']
            self.args.actor_lr = config['TD3']['actor_lr']
            self.args.critic_lr = config['TD3']['critic_lr']
            self.args.hidden_size = [config['TD3']['layer1_size'],config['TD3']['layer2_size']]
            self.args.hidden_sizes = [config['TD3']['layer1_size'],config['TD3']['layer2_size']]
            self.args.tau = config['TD3']['tau']
            self.args.gamma = config['TD3']['gamma']
            self.args.batch_size = config['TD3']['batch_size']
            self.args.policy_noise = config['TD3']['policy_noise']
            self.args.noise_clip = config['TD3']['noise_clip']
            self.args.policy_delay = config['TD3']['policy_delay']
            self.max_action = config['TD3']['max_action']

            self.args.buffer_size = config['Experiment'][self.mode]['buffer_memory_size']
            if self.ID == 'First':
                self.freeze_agent = self.config['TD3']['freeze_agent']
            elif self.ID == 'Second':
                self.freeze_agent = self.config['TD3']['freeze_second_agent']
            
            self.chkpt_dir = config['TD3']['chkpt']
            if self.ID == 'First':
                self.load_file = config['TD3']['load_file']
            elif self.ID == 'Second':
                self.load_file = config['TD3']['load_second_file'] 
        else:
            self.args.actor_lr = alpha
            self.args.critic_lr = beta
            self.args.hidden_size = [layer1_size,layer2_size]
            self.args.hidden_sizes = [layer1_size,layer2_size]
            self.args.tau = tau
            self.args.gamma = gamma
            self.args.batch_size = batch_size
            self.load_file = load_file
            self.chkpt_dir = chkpt_dir

        
       
        self.buffer_max_size = buffer_max_size
        self.env = env
        self.p_name = participant_name

        self.args.state_shape = input_dims[0] 
        self.args.action_shape  = args.num_actions
        #print("action_shape",self.args.action_shape)
        #print("state_shape",self.args.state_shape)
        
        self.model_name = config['TD3']['model_name']

        # Saving arrays
       
        
        self.q1_history = []
        self.q2_history = []

        self.q1_loss_history = []
        self.q2_loss_history = []

        self.policy_history = []
        self.policy_loss_history = []

      
      

        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.next_states_history = []
        self.dones_history = []
        self.transition_infos = []
        self.total_it = 0

        
        self.actor = Actor(self.args.state_shape, self.args.action_shape, self.args.hidden_sizes,name=self.model_name, chkpt_dir=self.chkpt_dir,load_file = self.load_file).to(device)
        self.actor_target = Actor(self.args.state_shape,self.args.action_shape,self.args.hidden_sizes,name=self.model_name + '_target',chkpt_dir=self.chkpt_dir,load_file=self.load_file).to(device)


        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.args.state_shape, self.args.action_shape, self.args.hidden_sizes,name=self.model_name, chkpt_dir=self.chkpt_dir,load_file = self.load_file).to(device)
        self.target_critic = Critic(self.args.state_shape, self.args.action_shape, self.args.hidden_sizes,name=self.model_name +'_target', chkpt_dir=self.chkpt_dir,load_file = self.load_file).to(
            device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr, eps=1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.critic_lr, eps=1e-8)

       
        self.memory = ReplayBuffer(self.args.buffer_size)
        assert self.memory is not None, "ReplayBuffer was not initialized."
        #print("ReplayBuffer iniialized")

        
        if self.args.Load_Expert_Buffers:
            self.memory.merge_buffers(self.args)

        if self.args.load_buffer:
            self.memory.load_buffer(self.args.buffer_path_1)
            
    def learn(self, cycle_i):
        
        self.total_it += 1 # delay
        self.update_num = 0  #not used

        #print(f"Current buffer size: {self.memory.get_size()}")
     
     

        states, actions, rewards, states_, dones, transition_info = self.memory.sample(self.args.batch_size,self.update_num)
        #print(f"States shape: {states.shape}")  
        #print(f"Actions shape: {actions.shape}")
     
    # Convert to tensors
        states = torch.from_numpy(states).float().to(device)
        states_ = torch.from_numpy(states_).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        dones = torch.from_numpy(dones).float().unsqueeze(1).to(device)
        
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).long().to(device)
        else:
            actions = actions.long().to(device)  

    #  one-hot encoding ->states[bs,state_dim] concatenated actions[batchsize,num_action]
        actions = F.one_hot(actions, num_classes=self.args.num_actions).float()
        
        num_actions = self.args.num_actions  
       
        sa = torch.cat([states, actions], dim=1)
        #print(f"Concatenated shape: {sa.shape}") 

    
        #with torch.no_grad():
            #continuous_action = self.actor_target(states_)
            #noise = torch.randn_like(continuous_action) * self.args.policy_noise
            #noisy_action = (continuous_action + noise).clip(-1, 1) 
            #discrete_action = torch.round(noisy_action) 
        with torch.no_grad():
            noise = (torch.randn_like(self.actor_target(states_)) * self.args.policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            next_actions = (self.actor_target(states_) + noise).clamp(-self.max_action, self.max_action)
            next_discrete_actions = torch.argmax(next_actions, dim=-1)
            next_discrete_actions_one_hot = F.one_hot(next_discrete_actions, num_classes=num_actions).float()
            #next_action_indices = torch.argmax(next_discrete_actions_one_hot, dim=-1, keepdim=True)

            target_q1, target_q2 = self.target_critic(states_, next_discrete_actions_one_hot)
            target_q = rewards + (1 - dones) * self.args.gamma * torch.min(target_q1, target_q2)#unsqueeze
            


        # Compute target Q-values
        #target_q1, target_q2 = self.target_critic(states_, discrete_actions)
        #target_q = rewards + (1 - dones) * self.args.gamma * torch.min(target_q1, target_q2)

    # Get current Q estimates
        current_q1, current_q2 = self.critic(states, actions)
        #current_q1, current_q2 = self.critic(sa)

    
        q1_loss = F.mse_loss(current_q1, target_q.detach())
        q2_loss = F.mse_loss(current_q2, target_q.detach())
        critic_loss = q1_loss + q2_loss

   
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

   
        if self.total_it % self.args.policy_delay == 0:
        # Compute actor loss
            q1 = self.critic.q1(states, self.actor(states))
            actor_loss = -q1.mean()
            

        
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        get networks
            self.soft_update(self.actor_target, self.actor, self.args.tau)
            self.soft_update(self.target_critic, self.critic, self.args.tau)

       
            self.policy_loss_history.append(actor_loss.item())
        else:
            actor_loss = 0.0  

    # Log critic losses
        self.q1_loss_history.append(q1_loss.item())
        self.q2_loss_history.append(q2_loss.item())
     
        self.update_num += 1

        return actor_loss, q1_loss.item(), q2_loss.item()


    def supervised_learn(self,states,actions,total_updates):
        # lists to tensors
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)

        states = torch.from_numpy(states).float().to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]

        for i in range(total_updates):
            batch_ids = np.random.choice(states.shape[0], self.args.batch_size)
            input_batch = states[batch_ids]
            truth_batch = actions[batch_ids]

            self.actor_optim.zero_grad()
            action_probs = self.actor(input_batch)
            #print(action_probs)
            loss = F.cross_entropy(action_probs, truth_batch.squeeze(1))
            loss.backward()
            self.actor_optim.step()
            


            print("Supervised learning loss: ", loss.item())


    
    
    def soft_update(self, target_net, source_net, tau):
       for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def soft_update_target(self):
        """
    Perform a soft update on both the actor and critic target networks.
    """
        self.soft_update(self.actor_target, self.actor, self.args.tau)
        self.soft_update(self.target_critic, self.critic, self.args.tau)


    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions)  # select the Q corresponding to chosen A
        curr_q2 = curr_q2.gather(1, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
          noise = (torch.randn_like(rewards) * self.args.policy_noise).clamp(
            -self.args.noise_clip, self.args.noise_clip)
          target_actions = (self.actor_target(next_states) + noise).clamp(
            -self.max_action, self.max_action)
        target_q1, target_q2 = self.target_critic(next_states, target_actions)
        target_q = rewards + (1 - dones) * self.args.gamma * torch.min(target_q1, target_q2)
        
        return target_q


 
    def calc_critic_loss(self, states, actions, rewards, next_states, dones):
        target_q = self.calc_target_q(rewards, next_states, dones)
        curr_q1, curr_q2 = self.calc_current_q(states, actions)
        q1_loss = F.mse_loss(curr_q1, target_q.detach())
        q2_loss = F.mse_loss(curr_q2, target_q.detach())
        critic_loss = q1_loss + q2_loss
       
        return critic_loss, q1_loss.item(), q2_loss.item()

    def calc_actor_loss(self, states):
        actor_loss = -self.critic.q1(states, self.actor(states)).mean()
        return actor_loss
    def calc_actor_loss(self, states):
        actions_pred = self.actor(states)
        actor_loss = -self.critic.q1(states, actions_pred).mean()
        return actor_loss



    

    def save_models(self,block_number):
        if self.chkpt_dir is not None:
            print('.... saving models ....')
            self.actor.save_checkpoint(block_number)
            self.critic.save_checkpoint(block_number)
            #self.target_critic.save_checkpoint()
    


    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.target_critic.load_checkpoint()

    
    def collect_data(self):
        return_data = {}

        # Collect data for plotting.
        return_data['q1'] = self.q1_history
        return_data['q2'] = self.q2_history

        return_data['policy'] = self.policy_history
        return_data['policy_loss'] = self.policy_loss_history

        return_data['q1_loss'] = self.q1_loss_history
        return_data['q2_loss'] = self.q2_loss_history
        
       
        return_data['states'] = self.states_history
        return_data['actions'] = self.actions_history
        return_data['rewards'] = self.rewards_history
        return_data['next_states'] = self.next_states_history
        return_data['dones'] = self.dones_history



       

        self.q1_history = []
        self.q2_history = []

        self.q1_loss_history = []
        self.q2_loss_history = []

        self.policy_history = []
        self.policy_loss_history = []

        self.entropy_history = []
        self.entropy_loss_history = []

        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.next_states_history = []
        self.dones_history = []

        return return_data
    
    def return_settings(self):
        actor_lr = self.args.actor_lr
        critic_lr = self.args.critic_lr
        hidden_size = self.args.hidden_size
        tau = self.args.tau
        gamma = self.args.gamma
        batch_size = self.args.batch_size
        policy_noise = self.args.policy_noise
        noise_clip = self.args.noise_clip
        policy_delay = self.args.policy_delay
        max_action = self.max_action
        freeze_status = self.freeze_agent

        return (
        self.ID, actor_lr, critic_lr, hidden_size, tau, gamma, batch_size,
        policy_noise, noise_clip, policy_delay, max_action, freeze_status
    )