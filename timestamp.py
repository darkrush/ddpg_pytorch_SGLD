import time
import argparse
import pickle
import numpy as np

class TimeStamp(object):
    def __init__(self):
        self.next_id = 0
        self.timeline = []
    def start(self,name):
        self.timeline.append((self.next_id, name, "start",time.time()))
        self.next_id+=1
    def end(self,name):
        self.timeline.append((self.next_id, name, "end"  ,time.time()))
        self.next_id+=1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TimeStamp parser')
    parser.add_argument('--p', type=str, help='pkl file path')
    parser.add_argument('--o', type=str, help='output file path')
    args = parser.parse_args()
    with open(args.p ,'rb') as f :
        timeline = pickle.load(f)
    
    name_stack = []
    item_dict = {}
    item_list = []
    
    
    for id,name,se,time in timeline:
        if se == 'start' :
            name_stack.append((name,time))
            if name not in item_dict:
                item_dict[name] = len(item_list)
                item_list.append((name,[]))
        elif se == 'end':
            assert name_stack[-1][0]==name
            assert item_list[item_dict[name]][0] == name
            item_list[item_dict[name]][1].append(time - name_stack[-1][1])
            name_stack.pop()
    assert len(name_stack) == 0
    
    with open(args.o ,'w') as f:
        for name,time_list in item_list:
            number = len(time_list)
            time_list = np.array(time_list).reshape(-1,1)
            print("{}\t{}\t{}\t{}".format(name,number,time_list.mean()*1000,time_list.std(ddof = 1)*1000))
            f.write("{}\t{}\t{}\t{}\n".format(name,number,time_list.mean()*1000,time_list.std(ddof = 1)*1000))