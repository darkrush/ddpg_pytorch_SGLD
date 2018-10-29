import numpy as np
import torch
import argparse
import gym.spaces
import gym
import time

from model import Actor
from utils import *

from multiprocessing import Process, Queue
from logger import Singleton_logger

class Evaluator(object):
    def __init__(self):
        self.env = None
        self.actor = None
        self.obs_norm = None
        self.visualize = False
        
    def set_up(self, env,num_episodes = 10, max_episode_length=None,load_dir = None, apply_norm = True, multi_process = True, logger = True,visualize = False, rand_seed = -1):
        self.env_name = env
        self.num_episodes = num_episodes
        self.max_episode_length = 1000
        self.load_dir = load_dir
        self.apply_norm = apply_norm
        self.multi_process = multi_process
        self.logger = logger
        self.visualize = visualize
        self.rand_seed = rand_seed
        if self.multi_process :
            self.queue = Queue(maxsize = 1)
            self.sub_process = Process(target = self.start_eval_process,args = (self.queue,))
            self.sub_process.start()
        else :
            self.env = gym.make(self.env_name)
            if self.rand_seed >= 0:
                self.env.seed(self.rand_seed)
            self.action_scale = (self.env.action_space.high - self.env.action_space.low)/2.0
            self.action_bias = (self.env.action_space.high + self.env.action_space.low)/2.0

        
    def load_module(self, buffer):
        self.actor = torch.load(buffer)
        
    def __load_actor(self,load_dir = None):
        if load_dir is None:
            load_dir = self.load_dir
        assert load_dir is not None
        self.actor = torch.load('{}/actor.pkl'.format(load_dir))
        if self.apply_norm:
            self.obs_norm = torch.load('{}/obs_norm.pkl'.format(output))
    
    def __get_action(self, observation):
        obs = torch.tensor([observation],dtype = torch.float32,requires_grad = False).cuda()
        if self.apply_norm :
            obs = self.obs_norm(obs)
            
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy().squeeze(0)
        action = np.clip(action, -1., 1.)
        return action * self.action_scale + self.action_bias
        
    def __run_eval(self,totoal_cycle):
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
                
                if self.visualize & (episode == 0):
                    self.env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        result_mean = result.mean()
        result_std = result.std(ddof = 1)
        if self.logger :
            Singleton_logger.trigger_log( 'eval_reward_mean',result_mean, totoal_cycle)
            Singleton_logger.trigger_log( 'eval_reward_std',result_std, totoal_cycle)
        localtime = time.asctime( time.localtime(time.time()) )
        print("{} eval : cycle {:<5d}\treward mean {:.2f}\treward std {:.2f}".format(localtime,totoal_cycle,result_mean,result_std))

    def load_and_run(self,totoal_cycle):
        self.__load_actor()
        self.__run_eval(totoal_cycle)
        
    def trigger_eval_process(self,totoal_cycle):
        if self.multi_process :
            self.queue.put(totoal_cycle,block = False)
        else :
            self.load_and_run(totoal_cycle)

    def trigger_close(self):
        if self.multi_process :
            self.queue.put(-1,block = True)
        
    def start_eval_process(self,queue):
        
        self.env = gym.make(self.env_name)
        if self.rand_seed >= 0:
            self.env.seed(self.rand_seed)
        self.action_scale = (self.env.action_space.high - self.env.action_space.low)/2.0
        self.action_bias = (self.env.action_space.high + self.env.action_space.low)/2.0
        
        while True:
            item = queue.get(block = True)
            if item < 0:
                break
            totoal_cycle = item
            self.load_and_run(totoal_cycle)

    def __del__(self):
        if self.env is not None:
            self.env.close()
    
    
Singleton_evaluator = Evaluator()

'''
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
'''