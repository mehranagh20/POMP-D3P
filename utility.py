import torch
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
import robosuite as suite
from robosuite.wrappers import GymWrapper


def xu2t(x, u, DIM_X, DIM_U):
    x = np.array([x]).reshape(-1, DIM_X)
    u = np.array([u]).reshape(-1, DIM_U)
    return torch.FloatTensor(np.concatenate([x, u], 1)).type(torch.FloatTensor)


def x2t(x, DIM_X):
    x = np.array([x]).reshape(-1, DIM_X)
    return torch.FloatTensor(x).type(torch.FloatTensor)


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DecayLR:
    def __init__(self, tmax=100000):
        self.tmax = int(tmax)
        assert self.tmax > 0
        self.lr_step = (0 - 1) / self.tmax

    def step(self, step):
        lr = 1 + self.lr_step * step
        lr = max(1e-6, min(1.0, lr))
        return lr


def get_lrschedule(args, optimizer):
    if args.lr_schedule:
        scheduler = DecayLR(tmax=args.num_steps)
        lr_scheduler = LambdaLR(optimizer, lambda x: scheduler.step(x))
    else:
        lr_scheduler = LambdaLR(optimizer, lambda x: 1.0)
    return lr_scheduler

def robosuite_env(env_name, robot, seed):
    env = GymWrapper(
    suite.make(
        env_name=env_name,  # try with other tasks like "Stack" and "Door"
        robots=robot,  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=200,
        use_object_obs=True,
    ))
    env._max_episode_steps = 200

        # reset the environment
    env.reset()
    return env
