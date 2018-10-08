

import matplotlib.pyplot as plt
import pickle
import numpy as np

result_dir = 'result/'
env_name = 'Humanoid-v2'
log_file = 'log_data_dict.pkl'
s = 1
t = 10
idx_list = [x for x in range(s,t+1)]
mean_array  = None

for idx in range(len(idx_list)):
    with open(result_dir+env_name+'-run{}/'.format(idx_list[idx])+log_file,'rb') as f:
        m=pickle.load(f)
    reward_mean,step = zip(*m['eval_reward_mean'])
    reward_mean = np.array(reward_mean)
    if mean_array is None:
        mean_array = reward_mean
    else:
        mean_array += reward_mean
    step = np.array(step)
mean_array/=(t-s+1)
plt.plot(step,mean_array,label= 'SGLD')

s = 11
t = 20
idx_list = [x for x in range(s,t+1)]
mean_array  = None

for idx in range(len(idx_list)):
    with open(result_dir+env_name+'-run{}/'.format(idx_list[idx])+log_file,'rb') as f:
        m=pickle.load(f)
    reward_mean,step = zip(*m['eval_reward_mean'])
    reward_mean = np.array(reward_mean)
    if mean_array is None:
        mean_array = reward_mean
    else:
        mean_array += reward_mean
    step = np.array(step)
mean_array/=(t-s+1)
plt.plot(step,mean_array,label= 'an')
plt.legend()
plt.show()
