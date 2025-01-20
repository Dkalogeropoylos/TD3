# Virtual environment
import numpy as np
from maze3D_new.Maze3DEnvRemote import Maze3D as Maze3D_v2
# from maze3D_new.assets import *
# from maze3D_new.utils import save_logs_and_plot

# Experiment
from game.experiment import Experiment
from game.game_utils import  get_config
# RL modules
from rl_models.utils import get_td3_agent

import sys
import time
from datetime import timedelta
import argparse
import torch
import os
from prettytable import PrettyTable
from datetime import timedelta

"""
The code of this work is based on the following github repos:
https://github.com/kengz/SLM-Lab
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--participant", type=str, default="test")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=3500)
    parser.add_argument("--actor-lr", type=float, default=0.0003)
    parser.add_argument("--critic-lr", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    
   
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=[32, 32])
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[32, 32])
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--num-actions", type=int, default=3)
    parser.add_argument("--agent-type", type=str, default="basetd3")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--policy-noise", type=float, default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy-delay", type=int, default=2, help="Frequency of delayed policy updates")
    parser.add_argument("--max-action", type=float, default=1.0, help="Maximum action value")

    parser.add_argument("--avg-q", action="store_true", default=False)
    parser.add_argument('--clip-q', action="store_true", default=False)
    parser.add_argument("--clip-q-epsilon", type=float, default=0.5)
    

    parser.add_argument('--buffer-path-1',type=str,default='game/Saved_Buffers/Buffer_Cris.npy')
    parser.add_argument('--buffer-path-2',type=str,default='game/Saved_Buffers/Buffer_Koutris.npy')
    parser.add_argument('--buffer-path-3',type=str,default=None)

    parser.add_argument('--Load-Expert-Buffers',action='store_true',default=False)

    parser.add_argument('--load-buffer',action='store_true',default=False)




    return parser.parse_args()

def print_setting(agent, x):
    # Get TD3-specific settings from the agent
    (
        ID, actor_lr, critic_lr, hidden_size, tau, gamma, batch_size,
        policy_noise, noise_clip, policy_delay, max_action, freeze_status
    ) = agent.return_settings()
    
    # Update the table headers to include TD3 parameters
    x.field_names = [
        "Agent ID", "Actor LR", "Critic LR", "Hidden Size", "Tau", "Gamma", "Batch Size",
        "Policy Noise", "Noise Clip", "Policy Delay", "Max Action", "Freeze Status"
    ]
    
    # Add the agent's settings to the table
    x.add_row([
        ID, actor_lr, critic_lr, hidden_size, tau, gamma, batch_size,
        policy_noise, noise_clip, policy_delay, max_action, freeze_status
    ])
    return x

def check_save_dir(config,participant_name):
    checkpoint_dir = config['TD3']['chkpt']
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if os.path.isdir(os.path.join(checkpoint_dir,participant_name)) == False:
        os.mkdir(os.path.join(checkpoint_dir,participant_name))
        checkpoint_dir = os.path.join(checkpoint_dir,participant_name) + '/'
    else:
        c = 1
        while os.path.isdir(os.path.join(checkpoint_dir,participant_name+str(c))) == True:
            c += 1
        os.mkdir(os.path.join(checkpoint_dir,participant_name+str(c)))
        checkpoint_dir = os.path.join(checkpoint_dir,participant_name+str(c)) + '/'

    config['TD3']['chkpt'] = checkpoint_dir
    return config

def main(argv):
    args = get_args()
    # get configuration
    print('IM trying to get this config')
    print(args.config)
    
    config = get_config(args.config)
    
    print('Config loaded',config)

    # creating environment
    maze = Maze3D_v2(config_file=args.config)
    loop = config['Experiment']['mode']
    if loop != 'human':
        config = check_save_dir(config,args.participant)
        print_array = PrettyTable()
        if args.agent_type == "basetd3":
            agent = get_td3_agent(args,config, maze,p_name=args.participant,ID='First')
            agent.save_models('Initial')
        print_array = print_setting(agent,print_array)

        if loop == 'no_tl_two_agents':
            if args.agent_type == "basetd3":
                second_agent = get_td3_agent(args,config, maze, p_name=args.participant,ID='Second')
                second_agent.save_models('Initial')
            print_array = print_setting(second_agent,print_array)
        else:
            second_agent = None

        print('Agent created')
        print(print_array)
        # create the experiment
    else:
        agent = None
        second_agent = None
    experiment = Experiment(maze, agent, config=config,participant_name=args.participant,second_agent=second_agent)

    start_experiment = time.time()

    # Run a Pre-Training with Expert Buffers
    print('Load Expert Buffers:',args.Load_Expert_Buffers)
    if args.Load_Expert_Buffers:
        experiment.test_buffer(2500)
    elif args.load_buffer:
        experiment.test_buffer(2500)



    if loop == 'no_tl':
        experiment.mz_experiment(args.participant)
    elif loop == 'no_tl_only_agent':
        experiment.mz_only_agent(args.participant)
    elif loop == 'no_tl_two_agents':
        experiment.mz_two_agents(args.participant)
    elif loop == 'eval':
        experiment.mz_eval(args.participant)
    elif loop == 'human':
        experiment.human_play(args.participant)
    else:
        print("Unknown training mode")
        exit(1)
    #experiment.env.finished()
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)
    
    print('Total Experiment time: {}'.format(experiment_duration))

    return


if __name__ == '__main__':
   
    main(sys.argv[1:])
    exit(0)