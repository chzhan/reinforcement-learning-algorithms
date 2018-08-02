import gym
from trpo_agent import trpo_agent
import mujoco_py
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    # build up the training environment
    env = gym.make(args.env_name)
    # start to train the environment
    trpo_trainer = trpo_agent(args, env)
    # train the network...
    trpo_trainer.train_network()


