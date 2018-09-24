import torch
import os

USE_CUDA = torch.cuda.is_available()


def to_numpy(var, use_cuda = None):
    if use_cuda is None:
        use_cuda = USE_CUDA
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=torch.float, use_cuda = None):
    if use_cuda is None:
        use_cuda = USE_CUDA
    tensor = torch.tensor(data = ndarray, dtype = dtype, requires_grad = requires_grad)
    if use_cuda : tensor = tensor.cuda()
    return tensor

def SGLD_update(net,lr,coef):
    scale = (lr*coef)**0.5
    for param in net.parameters():
        #param.data.copy_( param.data + scale*torch.randn(param.shape) )
        param.data.copy_( param.data + scale*torch.cuda.FloatTensor(param.shape).normal_())
      

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)        
        
def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir