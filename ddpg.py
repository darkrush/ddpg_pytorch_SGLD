import io
import numpy as np
import torch.nn as nn
from torch.optim import Adam,SGD
from agent_pool import Agent_pool

from obs_norm import Run_Normalizer
from model import Actor,Critic
from memory import Memory
from utils import *


class DDPG(object):
    def __init__(self, nb_actions, nb_states, layer_norm, obs_norm,
                 actor_lr, critic_lr,SGLD_coef,noise_decay,lr_decay, batch_size,
                 discount, tau, pool_size,
                 parameters_noise, action_noise):
                 
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.layer_norm = layer_norm
        self.parameters_noise = parameters_noise
        self.action_noise = action_noise
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.pool_size = pool_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.SGLD_coef = SGLD_coef
        self.noise_coef = 1
        self.noise_decay = noise_decay
        self.lr_coef = 1
        self.lr_decay = lr_decay

        self.actor = Actor(nb_states = self.nb_states, nb_actions = self.nb_actions, layer_norm = self.layer_norm)
        self.actor_target = Actor(nb_states = self.nb_states, nb_actions = self.nb_actions, layer_norm = self.layer_norm)
        #self.actor_optim  = SGD(self.actor.parameters(), lr=actor_lr, momentum=0.9,weight_decay  = 0.01)
        self.actor_optim  = Adam(self.actor.parameters(), lr=actor_lr)
        hard_update(self.actor_target,self.actor)
        
        self.critic = Critic(nb_states, nb_actions, layer_norm = self.layer_norm)
        self.critic_target = Critic(nb_states, nb_actions, layer_norm = self.layer_norm)
        #self.critic_optim  = SGD(self.critic.parameters(), lr=critic_lr, momentum=0.9,weight_decay  = 0.01)
        self.critic_optim  = Adam(self.critic.parameters(), lr=critic_lr)
        hard_update(self.critic_target,self.critic)
        
        
        self.memory = Memory(int(1e6), (nb_actions,), (nb_states,))
        self.obs_norm = obs_norm
        if self.obs_norm:
            self.run_obs_norm = Run_Normalizer((nb_states,))
        self.is_training = True
        
        if self.pool_size>0:
            self.agent_pool = Agent_pool(self.pool_size)
        
        self.s_t = None
        self.a_t = None
        
        
        if USE_CUDA:
            self.cuda()
        
        self.stamper = None
        
        
        
    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, s_t1, done_t)
        if self.obs_norm:
            self.run_obs_norm.observe(s_t)
        self.s_t = s_t1
        
    def update(self):
        # Sample batch
        if self.stamper is not None :self.stamper.start('batch_sample')
        batch = self.memory.sample(self.batch_size)
        if self.stamper is not None :self.stamper.end('batch_sample')
        if self.stamper is not None :self.stamper.start('q_target')
        tensor_obs0 = to_tensor(batch['obs0'])
        tensor_obs1 = to_tensor(batch['obs1'])
        if self.obs_norm:
            tensor_obs0 = self.run_obs_norm.normalize(tensor_obs0)
            tensor_obs1 = self.run_obs_norm.normalize(tensor_obs1)
        
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                tensor_obs1,
                self.actor_target(tensor_obs1),
            ])
        
            target_q_batch = to_tensor(batch['rewards']) + \
                self.discount*to_tensor(1-batch['terminals1'].astype(np.float))*next_q_values
        if self.stamper is not None :self.stamper.end('q_target')
        
        # Critic update
        if self.stamper is not None :self.stamper.start('c_train')
        if self.stamper is not None :self.stamper.start('c_zero')
        self.critic.zero_grad()
        if self.stamper is not None :self.stamper.end('c_zero')
        if self.stamper is not None :self.stamper.start('c_critic')
        q_batch = self.critic([tensor_obs0, to_tensor(batch['actions']) ])
        if self.stamper is not None :self.stamper.end('c_critic')
        if self.stamper is not None :self.stamper.start('c_BP')
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        if self.stamper is not None :self.stamper.end('c_BP')
        if self.stamper is not None :self.stamper.start('c_Adam')
        self.critic_optim.step()
        if self.stamper is not None :self.stamper.end('c_Adam')
        #SGLD_update(self.critic, self.critic_lr*self.lr_coef,self.SGLD_coef)
        if self.stamper is not None :self.stamper.end('c_train')
        
        
        
        # Actor update
        if self.stamper is not None :self.stamper.start('a_train')
        if self.stamper is not None :self.stamper.start('a_zero')
        self.actor.zero_grad()
        if self.stamper is not None :self.stamper.end('a_zero')
        if self.stamper is not None :self.stamper.start('a_critic')
        policy_loss = -self.critic([
            tensor_obs0,
            self.actor(tensor_obs0)
        ])
        if self.stamper is not None :self.stamper.end('a_critic')
        if self.stamper is not None :self.stamper.start('a_BP')
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        if self.stamper is not None :self.stamper.end('a_BP')
        if self.stamper is not None :self.stamper.start('a_Adam')
        self.actor_optim.step()
        if self.stamper is not None :self.stamper.end('a_Adam')
        if self.stamper is not None :self.stamper.start('a_SGLD')
        SGLD_update(self.actor, self.actor_lr*self.lr_coef,self.SGLD_coef)
        if self.stamper is not None :self.stamper.end('a_SGLD')
        if self.stamper is not None :self.stamper.end('a_train')
        
        # Target update
        if self.stamper is not None :self.stamper.start('soft_update')
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        if self.stamper is not None :self.stamper.end('soft_update')
        return value_loss.item(),policy_loss.item()

    def apply_lr_decay(self):
        if self.lr_decay > 0:
            self.lr_coef = self.lr_decay*self.lr_coef/(self.lr_coef+self.lr_decay)
            self.critic_optim.param_groups[0]['lr'] = self.critic_lr * self.lr_coef
        
    def apply_noise_decay(self):
        if self.noise_decay > 0:
            self.noise_coef = self.noise_decay*self.noise_coef/(self.noise_coef+self.noise_decay)
        
    def select_action(self,random = False, s_t = None, if_noise = True):
        if random :
            action = np.random.uniform(-1.,1.,self.nb_actions)
        else:
            if s_t is None: raise RuntimeError()
            tensor_state = to_tensor(np.array([s_t]))
            if self.obs_norm:
                tensor_state = self.run_obs_norm.normalize(tensor_state)
            action = to_numpy(self.actor(tensor_state)).squeeze(0)

            if if_noise & (self.action_noise is not None):
                action += self.is_training*max(self.noise_coef, 0)*self.action_noise()
        action = np.clip(action, -1., 1.)
        self.a_t = action
        return action
        

    
    def load_weights(self, output): 
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )
        
    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
    def get_actor_buffer(self):
        buffer = io.BytesIO()
        torch.save(self.actor, buffer)
        return buffer
        
    def get_norm_param(self):
        return self.run_obs_norm.mean.cpu(),self.run_obs_norm.var.cpu()
    
    def append_actor(self):
        self.agent_pool.actor_append(self.actor.state_dict(),self.actor_target.state_dict())
        
    def pick_actor(self):
        actor,target = self.agent_pool.get_actor()
        self.actor.load_state_dict(actor)
        self.actor_target.load_state_dict(target)
    
    def append_critic(self):
        self.agent_pool.critic_append(self.critic.state_dict(),self.critic_target.state_dict())
        
    def pick_critic(self):
        critic,target = self.agent_pool.get_critic()
        self.critic.load_state_dict(critic)
        self.critic_target.load_state_dict(target)
        
    def append_agent(self,mode = 3):
        if mode == 1:
            self.append_actor()
        elif mode ==2:
            self.append_critic()
        elif mode ==3:
            self.append_actor()
            self.append_critic()

    def pick_agent(self,mode = 3):
        if mode == 1:
            self.pick_actor()
        elif mode ==2:
            self.pick_critic()
        elif mode ==3:
            self.pick_actor()
            self.pick_critic()

        
    def reset(self, obs):
        self.s_t = obs
        if self.action_noise is not None:
            self.action_noise.reset()
    
    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()