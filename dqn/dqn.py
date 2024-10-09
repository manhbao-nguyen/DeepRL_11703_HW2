#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm



class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.queue = collections.deque(maxlen=memory_size) 
        self.batch_size = batch_size
        # END STUDENT SOLUTION
       

    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        idxs = np.random.randint(0, len(self.queue), self.batch_size)
        states, actions, rewards, next_states, done = zip(*[self.queue[idx] for idx in idxs])

        batch = np.stack(states), np.array(actions), np.array(rewards), np.stack(next_states), np.array(done)
        batch = tuple(map(torch.FloatTensor, batch))
        return batch
        # END STUDENT SOLUTION



    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.queue.append(transition)
        # END STUDENT SOLUTION
        



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.q_net = q_net_init().to(device)
        self.target = deepcopy(self.q_net).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = lr_q_net)

        self.buffer = ReplayMemory(memory_size=replay_buffer_size,batch_size=replay_buffer_batch_size)
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.q_net(state), self.target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        with torch.no_grad():

            q_values = self.q_net(torch.FloatTensor(state)).numpy()
        if stochastic and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size) 
        else:
            return np.argmax(q_values)
        # END STUDENT SOLUTION



    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION
        self.optimizer.zero_grad() 
        states, actions, rewards, next_states, done = self.buffer.sample_batch()
        target = rewards + self.gamma * (1 - done) * self.target(next_states).max(dim=-1)[0]
        pred_q = self.q_net(states)[torch.arange(len(actions)), actions.to(int)]
        loss = torch.nn.MSELoss()(pred_q, target)
        loss.backward()
        self.optimizer.step()
        # END STUDENT SOLUTION
 



    def run(self, env, max_steps, num_episodes, train, init_buffer = True):

        total_rewards = []


        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        # initialize buffer
        state = env.reset()[0]

        if init_buffer:
            for _ in range(self.burn_in):
                action = np.random.randint(self.action_size)

                next_state, r, done, _, _ = env.step(action)
                assert isinstance(state, np.ndarray) and isinstance(next_state, np.ndarray)
                self.buffer.append((state, action, r, next_state, done))
                if done:
                    state = env.reset()[0]
                else:
                    state = next_state

        steps = 0
        for episode in range(num_episodes):
            state = env.reset()[0]
            for t in tqdm(range(max_steps)):
                action = self.get_action(state=state, stochastic=True) 
                next_state, r, done, _, _ = env.step(action)
                done = done or (t == max_steps - 1)
                steps += 1
                assert isinstance(state, np.ndarray) and isinstance(next_state, np.ndarray)
                self.buffer.append((state, action, r, next_state, done))
                self.train()
                if steps % 50 == 0:
                    self.target = deepcopy(self.q_net)
                if done:
                    break
                state = next_state
            if (episode + 1) % 100 == 0:
                # testing 
                returns = [] 
                for _ in range(20):
                    state = env.reset()[0]
                    reward = 0 
                    for t in range(max_steps):
                        state, r, done, _, _ = env.step(self.get_action(state, False))
                        reward += r 
                        if done:
                            break 
                    returns.append(reward)

                total_rewards.append(np.mean(returns))

        # END STUDENT SOLUTION
        return(total_rewards)



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    trial_rewards = []
    for agent in agents:
        trial_rewards.append(agent.run(env, max_steps, num_episodes, train = True, init_buffer = True))
    trial_rewards =  np.array(trial_rewards)
    graph_every = 100
    average_total_rewards = trial_rewards.mean(0)
    min_total_rewards = trial_rewards.min(0)
    max_total_rewards = trial_rewards.max(0)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    agents = [DeepQNetwork(env.observation_space.shape[0], env.action_space.n) for _ in range(args.num_runs)]
    graph_agents("graph_0", agents,env, args.max_steps, args.num_episodes)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
