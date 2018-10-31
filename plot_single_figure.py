import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='DDPG on pytorch')
parser.add_argument('--logdir', type=str, help='result dir')
parser.add_argument('--low',type=int, help='first run number')
parser.add_argument('--high',type=int, help='last run number')
parser.add_argument('--log_file',default='log_data_dict.pkl', type=str, help='log pkl file')
parser.add_argument('--arg_file',default='args.txt', type=str, help='args txt file')
parser.add_argument('--smooth',default=0.7, type=float, help='curve smooth coef')

args = parser.parse_args()

for run_num in range(args.low,args.high+1):
    with open(os.path.join(args.logdir+'{}'.format(run_num), args.arg_file),'r') as f:
        argstr = f.read()
        argstr = argstr.split('(')[1]
        argstr = argstr.split(')')[0]
        argstr_list = argstr.split(',')
        args_dict = {argstr_item.split('=')[0].strip(): argstr_item.split('=')[1].strip() for argstr_item in argstr_list}
    with open(os.path.join(args.logdir+'{}'.format(run_num), args.log_file),'rb') as f:
        m=pickle.load(f)
        reward_mean,step = zip(*m['eval_reward_mean'])
        smooth_reward = [reward_mean[0],]
        for r in reward_mean:
            smooth_reward.append(smooth_reward[-1]*args.smooth + r*(1-args.smooth))
        smooth_reward=smooth_reward[1:]
        plt.plot(step,smooth_reward,label= args_dict['exp_name']+' SGLD_mode:'+args_dict['SGLD_mode']+' num_pseudo_batches:'+args_dict['num_pseudo_batches']+' pool_mode:'+args_dict['pool_mode'])
plt.legend()
plt.show()