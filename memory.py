import torch

class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit
        self.next_entry = 0
        self.nb_entries = 0
        
        self.data_buffer = {}
        self.data_buffer['obs0'] = torch.zeros((limit,) + observation_shape)
        self.data_buffer['obs1'] = torch.zeros((limit,) + observation_shape)
        self.data_buffer['actions'] = torch.zeros((limit,) + action_shape)
        self.data_buffer['rewards'] = torch.zeros((limit,1))
        self.data_buffer['terminals1'] = torch.zeros((limit,1))
    
    def getitem(self, idx):
        return {key: value[idx] for key,value in self.data_buffer.items()}
    
    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = torch.randint(0,self.nb_entries-1, (batch_size,),dtype = torch.long) 
        return {key: torch.index_select(value,0,batch_idxs) for key,value in self.data_buffer.items()}

    def reset(self):
        self.next_entry = 0
        self.nb_entries = 0
        
    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return
        
        self.data_buffer['obs0'][self.next_entry] = torch.from_numpy(obs0)
        self.data_buffer['obs1'][self.next_entry] = torch.from_numpy(obs1)
        self.data_buffer['actions'][self.next_entry] = torch.from_numpy(action)
        self.data_buffer['rewards'][self.next_entry] = torch.from_numpy(reward)
        self.data_buffer['terminals1'][self.next_entry] = torch.from_numpy(terminal1)
        
        if self.nb_entries < self.limit:
            self.nb_entries += 1
            
        self.next_entry = (self.next_entry + 1)%self.limit
        
        