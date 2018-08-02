from arguments import get_args
from ppo_agent import ppo_agent
import gym
import mujoco_py

if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name)
    ppo_tester = ppo_agent(env, args)
    ppo_tester.test_network()
