import io
import numpy as np
import torch.nn as nn
from torch.optim import Adam

from model import Actor,Critic
from memory import Memory
from utils import *


class DDPG(object):
    def __init__(self, nb_actions, nb_states, layer_norm,
                 actor_lr, critic_lr, batch_size,
                 discount, tau,
                 parameters_noise, action_noise):
                 
        self.nb_actions = nb_actions
        self.nb_states = nb_states
        self.layer_norm = layer_norm
        self.parameters_noise = parameters_noise
        self.action_noise = action_noise
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.epsilon = 1
        
        self.actor = Actor(nb_states = self.nb_states, nb_actions = self.nb_actions, layer_norm = self.layer_norm)
        self.actor_target = Actor(nb_states = self.nb_states, nb_actions = self.nb_actions, layer_norm = self.layer_norm)
        self.actor_optim  = Adam(self.actor.parameters(), lr=actor_lr)
        hard_update(self.actor_target,self.actor)
        
        self.critic = Critic(nb_states, nb_actions, layer_norm = self.layer_norm)
        self.critic_target = Critic(nb_states, nb_actions, layer_norm = self.layer_norm)
        self.critic_optim  = Adam(self.critic.parameters(), lr=critic_lr)
        hard_update(self.critic_target,self.critic)
        
        
        self.memory = Memory(int(1e6), (nb_actions,), (nb_states,))
        self.is_training = True
        
        self.s_t = None
        self.a_t = None
        
        
        if USE_CUDA:
            self.cuda()
        
        pass
        
        
    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, s_t1, done_t)
        self.s_t = s_t1
        
    def update(self):
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(batch['obs1']),
                self.actor_target(to_tensor(batch['obs1'])),
            ])
        
            target_q_batch = to_tensor(batch['rewards']) + \
                self.discount*to_tensor(1-batch['terminals1'].astype(np.float))*next_q_values
        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(batch['obs0']), to_tensor(batch['actions']) ])
        #print(q_batch.detach().cpu().numpy())
        #print(target_q_batch.detach().cpu().numpy())
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(batch['obs0']),
            self.actor(to_tensor(batch['obs0']))
        ])
        
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        return value_loss.item(),policy_loss.item()
        
    def select_action(self,random = False, s_t = None, if_noise = True):
        if random :
            action = np.random.uniform(-1.,1.,self.nb_actions)
        else:
            if s_t is None: raise RuntimeError()
            action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)

            if if_noise & (self.action_noise is not None):
                action += self.is_training*max(self.epsilon, 0.0001)*self.action_noise()
                self.epsilon*=0.99
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
        
    def reset(self, obs):
        self.s_t = obs
        self.action_noise.reset()
    
    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()