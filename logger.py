from tensorboardX import SummaryWriter
from utils import *
import pickle
import os

from multiprocessing import Process, Queue,Lock


class Logger(object):
    def __init__(self):
        self.log_data_dict = {}
        self.writer = None
        self.output_dir = None
        self.queue = Queue()
        self.lock = Lock()
    def set_up(self,output_dir,with_writer = True):
        self.output_dir = output_dir
        if with_writer:
            self.writer = SummaryWriter(output_dir)
        self.sub_process = Process(target = self.start_log,args = (self.queue,self.lock))
        self.sub_process.start()
        
    def add_scalar(self,name,y,x):
        if self.writer is not None:
            self.writer.add_scalar(name,y,x)
        if name not in self.log_data_dict:
            self.log_data_dict[name] = []
        self.log_data_dict[name].append([y,x])
        
    def save_dict(self):
        with open(os.path.join(self.output_dir,'log_data_dict.pkl'),'wb') as f:
            pickle.dump(self.log_data_dict,f)
            
    def trigger_close(self):
        self.queue.put( None ,block = True)
            
    def trigger_log(self,name,y,x):
        self.queue.put((name,y,x),block = True)
        
    def trigger_save(self):
        self.queue.put(('__save__',0,0),block = True)
        
    def start_log(self,queue,lock):
        while True:
            item = queue.get(block = True)
            if item is None:
                if self.writer is not None:
                    self.writer.close()
                break
            lock.acquire()
            name,y,x = item
            if name == '__save__':
                self.save_dict()
            else:
                self.add_scalar(name,y,x)
            lock.release()
            
Singleton_logger = Logger()