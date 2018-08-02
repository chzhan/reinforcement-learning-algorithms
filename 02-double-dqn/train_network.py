import gym
from ddqn_agent import ddqn_agent
from arguments import get_args
"""
This module was used to train the dqn network...

"""

if __name__ == '__main__':
    # achieve the arguments...
    args = get_args()
    # start to create the environment...
    env = gym.make(args.env)
    ddqn_trainer = ddqn_agent(args, env)
    # start to train the network...
    ddqn_trainer.train_network()
