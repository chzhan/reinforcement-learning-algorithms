import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
import random

def select_actions(pi, dist_type):
    if dist_type == 'gauss':
        mean, std = pi
        actions = Normal(mean, std).sample()
    elif dist_type == 'beta':
        alpha, beta = pi
        actions = Beta(alpha.detach().cpu(), beta.detach().cpu()).sample()

    return actions.detach().cpu().numpy()[0]

# convert to tensor...
def state_to_tensor(state, use_cuda):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
    return state_tensor.cuda() if use_cuda else state_tensor

# calculate returns...
def calculate_returns(network, data, tau, gamma, next_value):
    # calculate all of the predicted value....
    state = torch.cat([element[0] for element in data], 0)
    pred_value, _ = network(state)
    # cat next value..
    pred_value = torch.cat([pred_value, next_value], 0)
    pred_value = pred_value.detach().cpu().numpy().squeeze()
    # start to calculate the returns...
    gae = 0
    for step in reversed(range(len(data))):
        delta = data[step][1] + gamma * pred_value[step + 1] * data[step][3] - pred_value[step]
        gae = delta + gamma * tau * data[step][3] * gae
        data[step][1] = gae + pred_value[step]

    return data

# generate samples...
def sample_generator(data, batch_size):
    random.shuffle(data)
    # generate chunks..
    chunks = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    return chunks

def get_log_probs(pi, actions, dist_type):
    if dist_type == 'gauss':
        mean, std = pi
        log_prob = Normal(mean, std).log_prob(actions)
    elif dist_type == 'beta':
        alpha, beta = pi
        log_prob = Beta(alpha, beta).log_prob(actions)

    return log_prob

