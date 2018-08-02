import gym
from ddqn_agent import ddqn_agent
from arguments import get_args

"""
This module was used to run the demo of the DQN...

"""

if __name__ == '__main__':
    # achieve the arguments...
    args = get_args()
    # start to create the environment...
    env = gym.make(args.env)
    # start to define the class..
    ddqn_tester = ddqn_agent(args, env)
    ddqn_tester.test_network()
