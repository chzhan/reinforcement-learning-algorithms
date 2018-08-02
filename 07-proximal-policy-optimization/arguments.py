import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--env-name', type=str, default='Walker2d-v2', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=32, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=3e-4, help='learning rate of the algorithm')
    parse.add_argument('--epoch', type=int, default=10, help='the epoch during training')
    parse.add_argument('--num-steps', type=int, default=2048, help='the steps to collect samples')
    parse.add_argument('--value-loss-coef', type=float, default=1, help='the coefficient of value loss')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--total-frames', type=int, default=1000000, help='the total frames for training')
    parse.add_argument('--dist', type=str, default='gauss', help='the distributions for sampling actions')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--clip', type=float, default=0.2, help='the ratio clip param')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')

    args = parse.parse_args()

    return args

