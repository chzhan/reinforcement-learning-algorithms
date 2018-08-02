import argparse

"""
This module was used to generate the parameters

That will be used for training the TRPO

2018-04-03

Author: Tianhong Dai

"""

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of the RL')
    parse.add_argument('--policy-lr', type=float, default=0.0005, help='the learning rate of the policy network')
    parse.add_argument('--value-lr', type=float, default=0.0005, help='the learning rate of the value network')
    parse.add_argument('--batch-size', type=int, default=64, help='the batch size to update the network')
    parse.add_argument('--env-name', type=str, default='Walker2d-v2', help='the name of the environment')
    parse.add_argument('--display-interval', type=int, default=1, help='the interval display the training information')
    parse.add_argument('--save-interval', type=int, default=100, help='the interval save the training models')
    parse.add_argument('--max-timestep', type=int, default=1000, help='the maximum time-step of each episode')
    parse.add_argument('--l2-reg', type=float, default=0.001, help='the weight decay coefficient of the optimizer')
    parse.add_argument('--max-kl', type=float, default=0.01, help='the maximal kl-divergence that allowed')
    parse.add_argument('--damping', type=float, default=0.1, help='the damping coefficient')
    parse.add_argument('--value-update_step', type=int, default=10, help='the update step for the value network')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='dir to save the models')
    parse.add_argument('--total-iterations', type=int, default=500, help='total iterations')

    args = parse.parse_args()

    return args
