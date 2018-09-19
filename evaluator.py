import numpy as np
import torch
import argparse
import gym.spaces
import gym

from model import Actor
from utils import *


class Evaluator(object):
    def __init__(self, env,num_episodes = 10, max_episode_length=None,load_dir = None):
    
        self.env = env
        self.actor = None
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.action_scale = (env.action_space.high - env.action_space.low)/2.0
        self.action_bias = (env.action_space.high + env.action_space.low)/2.0
        self.load_dir = load_dir
        
    def build_actor(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3, layer_norm = False):
        self.actor = Actor(nb_states = nb_states, nb_actions = nb_actions, hidden1=hidden1, hidden2=hidden2, init_w=init_w, layer_norm = layer_norm)
        
    def load_module(self, buffer):
        # load module(not only parameter, do not need build_actor before)
        self.actor = torch.load(buffer,map_location=torch.device('cpu'))
        
    def load_actor(self,load_dir = None):
        # load module parameter(only parameter, need build_actor before)
        if load_dir is None:
            load_dir = self.load_dir
        assert load_dir is not None
        self.actor = load_state_dict(
            torch.load('{}/actor.pkl'.format(load_dir))
        )

    def __get_action(self, observation):
        action = to_numpy(self.actor(to_tensor(np.array([observation]),use_cuda = False))).squeeze(0)
        action = np.clip(action, -1., 1.)
        return action * self.action_scale + self.action_bias
        
    def __call__(self, visualize=False):
        assert self.actor is not None
        observation = None
        result = []
        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = self.env.reset()
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = self.__get_action(observation)
                observation, reward, done, info = self.env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize & (episode == 0):
                    self.env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        return result.mean(),result.var()
    
    def __del__(self):
        self.env.close()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Eval DDPG')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--output', default='result/', type=str, help='result output dir')
    parser.add_argument('--max-episode-length', default=500, type=int, help='')
    
    args = parser.parse_args()
    output_dir = get_output_folder(args.output, args.env)
    env = gym.make(args.env)
    nb_actions = env.action_space.shape[0]
    nb_states = env.observation_space.shape[0]
    
    eval = Evaluator(env,num_episodes = 1, max_episode_length=None, load_dir = output_dir)
    eval.build_actor(nb_states, nb_actions)
    mean,var = eval(visualize = True)
    print(mean,var)