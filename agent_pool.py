import torch
class Agent_pool(object):
    def __init__(self,size = 10):
        self.size = size
        
        self.actor_next_id = 0
        self.actor_item_nb = 0
        self.actor_buffer = self.size*[[]]
        self.target_actor_buffer = self.size*[[]]
        
        self.critic_next_id = 0
        self.critic_item_nb = 0
        self.critic_buffer = self.size*[[]]
        self.target_critic_buffer = self.size*[[]]
        
    def actor_append(self,actor_state_dict,target_actor_state_dict):
        self.actor_buffer[self.actor_next_id ] = actor_state_dict
        self.target_actor_buffer[self.actor_next_id ] = target_actor_state_dict
        self.actor_next_id = (self.actor_next_id+1) % self.size
        if self.actor_item_nb < self.size:
            self.actor_item_nb = self.actor_item_nb+1
        return self.actor_next_id
        
    def critic_append(self,critic_state_dict,target_critic_state_dict):
        self.critic_buffer[self.critic_next_id ] = critic_state_dict
        self.target_critic_buffer[self.critic_next_id ] = target_critic_state_dict
        self.critic_next_id = (self.critic_next_id+1) % self.size
        if self.critic_item_nb < self.size:
            self.critic_item_nb = self.critic_item_nb+1
        return self.critic_next_id
    
    def agent_append(self,actor_state_dict,target_actor_state_dict,critic_state_dict,target_critic_state_dict):
        assert self.actor_next_id == self.critic_next_id 
        actor_next_id = self.actor_append(actor_state_dict,target_actor_state_dict)
        critic_next_id = self.critic_append(critic_state_dict,target_critic_state_dict)
        assert actor_next_id==critic_next_id
        return actor_next_id
        
    
    def get_actor(self,if_random = True, id = None):
        if if_random :
            id = torch.randint(0,self.actor_item_nb,[1]).item()
        else:
            if id is None:
                id = (self.actor_next_id-1)%self.size
        return self.actor_buffer[int(id)],self.target_actor_buffer[int(id)]

    def get_critic(self,if_random = True, id = None):
        if if_random :
            id = torch.randint(0,self.critic_item_nb,[1]).item()
        else:
            if id is None:
                id = (self.critic_next_id-1)%self.size
        return self.critic_buffer[id],self.target_critic_buffer[id]
        
    def get_agent(self,if_random = True, id = None):
        if if_random :
            id = torch.randint(0,self.critic_item_nb,[1]).item()
        else:
            if id is None:
                id = (self.critic_next_id-1)%self.size
        return self.actor_buffer[int(id)],self.target_actor_buffer[int(id)],self.critic_buffer[int(id)],self.target_critic_buffer[int(id)]
        
        
        
        
        
        
        