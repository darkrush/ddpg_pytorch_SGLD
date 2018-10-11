from tensorboardX import SummaryWriter
from utils import *
import pickle
import os

class Logger(object):
    def __init__(self):
        self.log_data_dict = {}
        self.writer = None
        self.output_dir = None
        
    def set_dir(self,output_dir,with_writer = True):
        self.output_dir = output_dir
        if with_writer:
            self.writer = SummaryWriter(output_dir)
        
    def add_scalar(self,name,y,x):
        if self.writer is not None:
            self.writer.add_scalar(name,y,x)
        if name not in self.log_data_dict:
            self.log_data_dict[name] = []
        self.log_data_dict[name].append([y,x])
        
    def save_dict(self):
        with open(os.path.join(self.output_dir,'log_data_dict.pkl'),'wb') as f:
            pickle.dump(self.log_data_dict,f)
    def close(self):
        if self.writer is not None:
            self.writer.close()

Singleton_logger = Logger()