from arguments import get_args
import gym
import models
import mujoco_py
from trpo_agent import trpo_agent

"""
This module was used to test the TRPO algorithms...
"""

if __name__ == '__main__':
    # get the arguments
    args = get_args()
    # set up the testing environment
    env = gym.make(args.env_name)
    # start to test...
    trpo_tester = trpo_agent(args, env)
    trpo_tester.test_network()

