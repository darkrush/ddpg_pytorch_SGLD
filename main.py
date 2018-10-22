from multiprocessing import Process,Queue
import numpy as np
import argparse
import io
import os
import pickle
from copy import deepcopy
import torch
import gym

from logger import Singleton_logger
from evaluator import Singleton_evaluator
from ddpg import DDPG
from noise import *
from utils import *


def train( agent, env, nb_epoch,nb_cycles_per_epoch,nb_rollout_steps,nb_train_steps, warmup, output_dir, obs_norm, max_episode_length=None, pool_mode = 0 ):

    action_scale = (env.action_space.high - env.action_space.low)/2.0
    action_bias = (env.action_space.high + env.action_space.low)/2.0
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = deepcopy(env.reset())
    agent.reset(observation)
    agent.append_agent(pool_mode)
        
    for t_warmup in range(warmup):
        action = agent.select_action(random = True)
        observation2, reward, done, info = env.step(action * action_scale + action_bias)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True
        # agent observe and update policy
        agent.store_transition(observation, action,np.array([reward,]), observation2, np.array([done,],dtype = np.float32))
        observation = deepcopy(observation2)
        if done: # end of episode
            # reset
            observation = deepcopy(env.reset())
            agent.reset(observation)
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    totoal_cycle = 0
    train_episode_reward = 0
    for epoch in range(nb_epoch):
        for cycle in range(nb_cycles_per_epoch):
            totoal_cycle+=1
            #pick actor
            agent.pick_agent(pool_mode)
            for t_rollout in range(nb_rollout_steps):        
                # agent pick action ...
                action = agent.select_action(random = False, s_t = [observation], if_noise = True)
                
                # env response with next_observation, reward, terminate_info
                observation2, reward, done, info = env.step(action * action_scale + action_bias)
                observation2 = deepcopy(observation2)
                if max_episode_length and episode_steps >= max_episode_length -1:
                    done = True
                
                # agent observe and update policy
                agent.store_transition(observation, action, np.array([reward,]), observation2, np.array([done,],dtype = np.float32))
            # update 
                step += 1
                episode_steps += 1
                episode_reward += reward
                observation = deepcopy(observation2)
            
                if done: # end of episode
                    # reset
                    observation = deepcopy(env.reset())
                    episode_steps = 0
                    train_episode_reward = episode_reward
                    episode_reward = 0.
                    episode += 1
            cl_list = []
            al_list = []
            for t_train in range(nb_train_steps):
                cl,al = agent.update()
                cl_list.append(cl)
                al_list.append(al)
            al_mean = np.mean(al_list)
            cl_mean = np.mean(cl_list)
            Singleton_logger.trigger_log('train_episode_reward',train_episode_reward, totoal_cycle)
            Singleton_logger.trigger_log( 'actor_loss_mean', al_mean, totoal_cycle)
            Singleton_logger.trigger_log( 'critic_loss_mean', cl_mean, totoal_cycle)

            agent.append_agent(pool_mode)

            
        agent.apply_noise_decay()
        agent.apply_lr_decay()
        agent.save_model(output_dir)
        
        Singleton_evaluator.trigger_eval_process(totoal_cycle = totoal_cycle)

        Singleton_logger.trigger_save()
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DDPG on pytorch')
    
    parser.add_argument('--env', default='HalfCheetah-v2', type=str, help='open-ai gym environment')
    parser.add_argument('--rand_seed', default=-1, type=int, help='random_seed')
    parser.add_argument('--nocuda', dest='with_cuda', action='store_false',help='disable cuda')
    parser.set_defaults(with_cuda=True)
    parser.add_argument('--actor-lr', default=0.0001, type=float, help='actor net learning rate')
    parser.add_argument('--critic-lr', default=0.001, type=float, help='critic net learning rate')
    parser.add_argument('--SGLD-coef', default=0.0001, type=float, help='critic net learning rate')
    parser.add_argument('--action-noise', dest='action_noise', action='store_true',help='enable action space noise')
    parser.set_defaults(action_noise=False)
    parser.add_argument('--noise-decay', default=0, type=float, help='action noise decay')
    parser.add_argument('--lr-decay', default=0, type=float, help='critic lr decay')
    parser.add_argument('--batch-size', default=128, type=int, help='minibatch size')
    parser.add_argument('--discount', default=0.99, type=float, help='reward discout')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--output', default='result/', type=str, help='result output dir')
    parser.add_argument('--stddev', default=0.2, type=float, help='action noise stddev')
    parser.add_argument('--obs-norm', dest='obs_norm', action='store_true',help='enable observation normalization')
    parser.set_defaults(obs_norm=False)
    parser.add_argument('--eval-visualize', dest='eval_visualize', action='store_true',help='enable render in evaluation progress')
    parser.set_defaults(eval_visualize=False)
    parser.add_argument('--nb-epoch', default=250, type=int, help='number of epochs')
    parser.add_argument('--nb-cycles-per-epoch', default=20, type=int, help='number of cycles per epoch')
    parser.add_argument('--nb-rollout-steps', default=100, type=int, help='number rollout steps')
    parser.add_argument('--nb-train-steps', default=50, type=int, help='number train steps')
    parser.add_argument('--max-episode-length', default=1000, type=int, help='max steps in one episode')
    parser.add_argument('--pool-size', default=10, type=int, help='agent pool size, 0 means no agent pool')
    parser.add_argument('--pool-mode', default=0, type=int, help='agent pool mode, 0: no pool, 1: actor pool only, 2: critic pool only, 3: both actor & critic')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        args.with_cuda = False
    if args.rand_seed >= 0 :
        if args.with_cuda:
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        np.random.seed(args.rand_seed)

    output_dir = get_output_folder(args.output, args.env)
    
    Singleton_logger.set_up(output_dir)
    Singleton_evaluator.set_up(args.env,num_episodes = 10, max_episode_length=args.max_episode_length,load_dir = output_dir, apply_norm = args.obs_norm,visualize = args.eval_visualize)
    
    eval_process = None
    
    
    with open(os.path.join(output_dir,'args.txt'),'w') as f:
        print(args,file = f)
        
    env = gym.make(args.env)
    if args.rand_seed >= 0 :
        env.seed(args.rand_seed)
        
    nb_actions = env.action_space.shape[0]
    nb_states = env.observation_space.shape[0]
    action_noise = None
    
    
    
    
    if args.action_noise:
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(args.stddev) * np.ones(nb_actions))

    
    agent = DDPG(nb_actions = nb_actions,nb_states = nb_states, layer_norm = True, obs_norm = args.obs_norm,
                 actor_lr = args.actor_lr, critic_lr = args.critic_lr,SGLD_coef = args.SGLD_coef,noise_decay = args.noise_decay,lr_decay = args.lr_decay, batch_size = args.batch_size,
                 discount = args.discount, tau = args.tau, pool_size = args.pool_size,
                 parameters_noise = None, action_noise = action_noise, with_cuda = args.with_cuda)
     
    train(agent = agent, env = env,
          nb_epoch = args.nb_epoch, nb_cycles_per_epoch =  args.nb_cycles_per_epoch, nb_rollout_steps =  args.nb_rollout_steps, nb_train_steps = args.nb_train_steps,
          warmup = args.warmup,output_dir = output_dir, obs_norm = args.obs_norm, max_episode_length=args.max_episode_length,pool_mode = args.pool_mode)
          
    Singleton_evaluator.trigger_close()
    Singleton_logger.trigger_close()
