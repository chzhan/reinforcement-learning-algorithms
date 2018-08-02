import gym
from dqn_agent import dqn_agent
from arguments import get_args
"""
This module was used to train the dqn network...

"""
if __name__ == '__main__':
    # achieve the arguments...
    args = get_args()
    # start to create the environment...
    env = gym.make(args.env)
    dqn_trainer = dqn_agent(args, env)
    # start to train the network...
    dqn_trainer.train_network()
