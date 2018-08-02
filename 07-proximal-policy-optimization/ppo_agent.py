import numpy as np
import torch
from torch import optim
from models import Network
from running_state import ZFilter
from utils import select_actions, state_to_tensor, calculate_returns, sample_generator, get_log_probs
from datetime import datetime
import os

class ppo_agent:
    def __init__(self, env, args):
        self.env = env 
        self.args = args
        # set the seeds..
        self.env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        # get the num of actions
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        # define the newtork...
        self.net = Network(num_states, num_actions, self.args.dist)
        self.old_net = Network(num_states, num_actions, self.args.dist)
        # if use the cuda...
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        # running filter...
        self.running_state = ZFilter((num_states,), clip=5)
        # check saving folder..
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # env folder..
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # start to train the network...
    def train_network(self):
        num_updates = self.args.total_frames // self.args.num_steps
        state = state_to_tensor(self.running_state(self.env.reset()), self.args.cuda)
        final_reward = 0
        for i in range(num_updates):
            brain_memory, total_reward = [], 0
            for steps in range(self.args.num_steps):
                with torch.no_grad():
                    _, pi = self.net(state)
                # select actions
                actions = select_actions(pi, self.args.dist)
                if self.args.dist == 'gauss':
                    input_actions = actions
                elif self.args.dist == 'beta':
                    input_actions = -1 + 2 * actions
                # take actions...
                state_, reward, done, _ = self.env.step(input_actions)
                mask = 0 if done else 1
                brain_memory.append([state, reward, actions, mask])
                # get rewards...
                total_reward += reward
                final_reward *= mask
                final_reward += (1 - mask) * total_reward
                total_reward *= mask
                state = state_to_tensor(self.running_state(self.env.reset() if done else state_), self.args.cuda)
            # get the next state_value...
            next_value, _ = self.net(state)
            brain_memory = calculate_returns(self.net, brain_memory, self.args.tau, self.args.gamma, next_value)
            print('[{}] Update: {} / {}, Frames: {}, Reward: {}'.format(datetime.now(), i, num_updates, \
                                                                            (i+1)*self.args.num_steps, final_reward))
            torch.save([self.net.state_dict(), self.running_state], self.model_path + '/model.pt')
            # load the old model...
            self.old_net.load_state_dict(self.net.state_dict())
            self._update_network(brain_memory)

    # update the entire network...
    def _update_network(self, data):
        for _ in range(self.args.epoch):
            collections = sample_generator(data, self.args.batch_size)
            for mini_batch in collections:
                state = torch.cat([element[0] for element in mini_batch], 0)
                returns = torch.tensor([element[1] for element in mini_batch], dtype=torch.float32).unsqueeze(1)
                actions = torch.tensor([element[2] for element in mini_batch], dtype=torch.float32)
                # decide if use cuda...
                if self.args.cuda:
                    returns = returns.cuda()
                    actions = actions.cuda()
                # get the old state value and policy
                with torch.no_grad():
                    value_old, pi_old = self.old_net(state)
                # advantages...
                advantages = (returns - value_old).detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                # get the current state value and policy
                value, pi = self.net(state)
                # get the log_prob
                old_log_prob = get_log_probs(pi_old, actions, self.args.dist).detach()
                log_prob = get_log_probs(pi, actions, self.args.dist)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr-1
                surr1 = prob_ratio * advantages
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                # get the value loss...
                value_loss = (returns - value).pow(2).mean()
                total_loss = policy_loss + self.args.value_loss_coef * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    # this part is to test the network
    def test_network(self):
        # load the model
        net_model, filter_model = torch.load(self.model_path+'/model.pt', map_location=lambda storage, loc: storage)
        self.net.load_state_dict(net_model)
        self.net.eval()
        # start to do the test
        for _ in range(1):
            state = self.env.reset()
            state = self._test_filter(state, filter_model.rs.mean, filter_model.rs.std)
            reward_sum = 0
            for _ in range(10000):
                self.env.render()
                state = state_to_tensor(state, self.args.cuda)
                with torch.no_grad():
                    _, pi = self.net(state)
                if self.args.dist == 'gauss':
                    mean, _ = pi
                    action = mean.detach().cpu().numpy().squeeze()
                elif self.args.dist == 'beta':
                    alpha, beta = pi
                    # calculate the mode
                    mode = (alpha - 1) / (alpha + beta - 2)
                    action = mode.detach().cpu().numpy().squeeze()
                state_, reward, done, _ = self.env.step(action)
                state_ = self._test_filter(state_, filter_model.rs.mean, filter_model.rs.std)
                reward_sum += reward
                if done:
                    break
                state = state_
            print('The reward of this episode is {}'.format(reward_sum))

    def _test_filter(self, x, mean, std, clip=10):
        x = x - mean
        x = x / (std + 1e-8)
        x = np.clip(x, -clip, clip)

        return x
