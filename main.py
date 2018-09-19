import numpy as np
import argparse
import io
from copy import deepcopy
import torch
import gym

from ddpg import DDPG
from evaluator import Evaluator
from noise import *
from utils import *

def train( agent, env, nb_epoch,nb_cycles_per_epoch,nb_rollout_steps,nb_train_steps, warmup, output_dir, max_episode_length=None):

    evaluator = Evaluator(env,num_episodes = 10, max_episode_length = max_episode_length ,load_dir = output_dir)
    
    action_scale = (env.action_space.high - env.action_space.low)/2.0
    action_bias = (env.action_space.high + env.action_space.low)/2.0

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = deepcopy(env.reset())
    agent.reset(observation)
    
    for t_warmup in range(warmup):
        action = agent.select_action(random = True)
        observation2, reward, done, info = env.step(action * action_scale + action_bias)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True
        # agent observe and update policy
        agent.store_transition(observation, action, reward, observation2, done)
        observation = deepcopy(observation2)
        if done: # end of episode
            # reset
            observation = deepcopy(env.reset())
            agent.reset()
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            
    for epoch in range(nb_epoch):
        print('epoch {}'.format(epoch))
        for cycle in range(nb_cycles_per_epoch):
            print('cycle {}'.format(cycle))
            for t_rollout in range(nb_rollout_steps):        
                # agent pick action ...
                action = agent.select_action(random = False, s_t = observation)
                # env response with next_observation, reward, terminate_info
                observation2, reward, done, info = env.step(action * action_scale + action_bias)
                observation2 = deepcopy(observation2)
                if max_episode_length and episode_steps >= max_episode_length -1:
                    done = True
                
                # agent observe and update policy
                agent.store_transition(observation, action, reward, observation2, done)
                           
            # update 
                step += 1
                episode_steps += 1
                episode_reward += reward
                observation = deepcopy(observation2)
            
                if done: # end of episode
                    # reset
                    observation = deepcopy(env.reset())
                    #print('train episode return {}'.format(episode_reward))
                    episode_steps = 0
                    episode_reward = 0.
                    episode += 1
            cl_list = []
            al_list = []
            for t_train in range(nb_train_steps):
                cl,al = agent.update()
                cl_list.append(cl)
                al_list.append(al)
            
        evaluator.load_module(io.BytesIO(agent.get_actor_buffer().getvalue()))
        eval_reward = evaluator(visualize = True)
        print(eval_reward)
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DDPG on pytorch')
    
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--actor-lr', default=0.0001, type=float, help='actor net learning rate')
    parser.add_argument('--critic-lr', default=0.001, type=float, help='critic net learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='minibatch size')
    parser.add_argument('--discount', default=0.99, type=float, help='reward discout')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--output', default='result/', type=str, help='result output dir')
    parser.add_argument('--stddev', default=0.2, type=float, help='action noise stddev')
    parser.add_argument('--nb-epoch', default=500, type=int, help='number of epochs')
    parser.add_argument('--nb-cycles-per-epoch', default=20, type=int, help='number of cycles per epoch')
    parser.add_argument('--nb-rollout-steps', default=100, type=int, help='number rollout steps')
    parser.add_argument('--nb-train-steps', default=50, type=int, help='number train steps')
    parser.add_argument('--max-episode-length', default=500, type=int, help='')
    
    args = parser.parse_args()
    output_dir = get_output_folder(args.output, args.env)

    env = gym.make(args.env)
    nb_actions = env.action_space.shape[0]
    nb_states = env.observation_space.shape[0]
    
    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(args.stddev) * np.ones(nb_actions))
    
    agent = DDPG(nb_actions = nb_actions,nb_states = nb_states, layer_norm = False,
                 actor_lr = args.actor_lr, critic_lr = args.critic_lr, batch_size = args.batch_size,
                 discount = args.discount, tau = args.tau,
                 parameters_noise = None, action_noise = action_noise)
     
    train(agent = agent, env = env,
          nb_epoch = args.nb_epoch, nb_cycles_per_epoch =  args.nb_cycles_per_epoch, nb_rollout_steps =  args.nb_rollout_steps, nb_train_steps = args.nb_train_steps,
          warmup = args.warmup,output_dir = output_dir, max_episode_length=args.max_episode_length)
