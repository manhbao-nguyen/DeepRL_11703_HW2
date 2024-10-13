import torch
from torch import nn, optim
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
import argparse
class PolicyGradient(nn.Module):
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, mode='REINFORCE', n=128, gamma=0.99, device='mps', **kwargs):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.kwargs = kwargs
        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            nn.Softmax(dim=-1)
            # END STUDENT SOLUTION
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, 1)
            # END STUDENT SOLUTION
        )

        # BEGIN STUDENT SOLUTION
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor.to(self.device)
        self.critic.to(self.device)

        max_steps = kwargs['max_steps']

        to_maxsteps = torch.arange(max_steps, device=self.device)
        grid = (to_maxsteps[:, None] - to_maxsteps[None,  :])
        self.N = kwargs['N'] 
        self.N = self.N if self.N is not None else max_steps

        grid_bool = torch.where((grid >= 0) & (grid < self.N), 1, 0)
        grid = grid_bool * grid
        self.gamma_matrix = (self.gamma ** grid) * grid_bool
        self.gamma_N = self.gamma ** self.N
        
        # END STUDENT SOLUTION


    def forward(self, state):
        return (self.actor(state), self.critic(state))


    def get_action(self, state, stochastic):
        # BEGIN STUDENT SOLUTION
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action_probs = action_probs.squeeze(0)

        if stochastic:
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
        else:
            action = torch.argmax(action_probs)

        return action.item()
        # END STUDENT SOLUTION


    def calculate_n_step_bootstrap(self, rewards):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        T = len(rewards)
        return (rewards[None, :] @ self.gamma_matrix[:T, :T])[0]
    

        # END STUDENT SOLUTION

    def calculate_n_step_a2c(self, rewards_tensor, values_tensor):
        T = len(rewards_tensor)
        returns = torch.zeros(T).to(self.device)
        N = self.N if self.N < T else T
        values_tensor = values_tensor[self.N:]
        
        values_tensor = torch.cat((values_tensor, torch.zeros(N).to(self.device)), 0)
        V_end = torch.where(torch.arange(T, device=self.device) + self.N < T, values_tensor, 0.0)
        returns = (rewards_tensor[None, :] @ self.gamma_matrix[:T, :T])[0] + self.gamma_N * V_end
        return returns

    def train(self, states, actions, rewards):
        # BEGIN STUDENT SOLUTION
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        if self.mode == 'REINFORCE':
            returns = self.calculate_n_step_bootstrap(rewards_tensor)
        elif self.mode == 'BASELINE':
            returns = self.calculate_n_step_bootstrap(rewards_tensor)
            baselines = self.critic(states_tensor)
            returns = returns - baselines
        else:
            values_tensor = self.critic(states_tensor).squeeze(1)
            returns = self.calculate_n_step_a2c(rewards_tensor, values_tensor)
            returns = returns.detach() - values_tensor

        
        action_probs = self.actor(states_tensor)
        
        action_distribution = torch.distributions.Categorical(action_probs)
        log_probs = action_distribution.log_prob(actions_tensor)

        policy_loss = - (log_probs * returns.detach()).mean()

        if self.mode != 'REINFORCE':
            actor_loss = (returns ** 2).mean()
            self.optimizer_critic.zero_grad()
            actor_loss.backward()
            self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()
        # END STUDENT SOLUTION


    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            state = env.reset()[0]
            states = []
            actions = []
            rewards = []
            total_reward = 0
            for t in range(max_steps):
                action = self.get_action(state, stochastic=True)
                # print(env.step(action))
                next_state, reward, done, _, _= env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward
                state = next_state

                if done:
                    break
            
            total_rewards.append(total_reward)
            if train:
                self.train(states, actions, rewards)
        # END STUDENT SOLUTION

        return(total_rewards)
    
def graph_agents(graph_name, agents, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    # Ensure that 'agents' is a list of D matrices from each trial
    # Each D matrix is assumed to be a NumPy array of shape (num_snapshots,)
    D = np.array(agents)  # Shape: (num_trials, num_snapshots)

    # Check if D has the correct shape
    if len(D.shape) != 2:
        raise ValueError("Each agent's data should be a 1D NumPy array representing snapshots.")

    num_trials, num_snapshots = D.shape

    average_total_rewards = np.mean(D, axis=0)  # Shape: (num_snapshots,)
    min_total_rewards = np.min(D, axis=0)       # Shape: (num_snapshots,)
    max_total_rewards = np.max(D, axis=0)       # Shape: (num_snapshots,)

    # Determine the episode number corresponding to each snapshot
    graph_every = num_episodes // num_snapshots  # e.g., 3500 / 35 = 100
    xs = [graph_every * (i + 1) for i in range(num_snapshots)]  # [100, 200, ..., 3500]

    output_dir = './graphs/'
    os.makedirs(output_dir, exist_ok=True)

    # END STUDENT SOLUTION

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(xs, min_total_rewards, max_total_rewards, color='skyblue', alpha=0.4, label='Min-Max Range')
    
    ax.plot(xs, average_total_rewards, color='blue', label='Mean Return')
    
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    
    ax.set_title(graph_name, fontsize=14)
    ax.set_xlabel('Training Episodes', fontsize=12)
    ax.set_ylabel('Average Total Reward', fontsize=12)
    
    ax.legend()
    
    # Adding grid for better readability
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, f'{graph_name}.pdf'))
    
    # Close the plot to free memory
    plt.close(fig)
    
    print(f'Finished: {graph_name}')
def run(env_name, max_steps, mode, num_trials, num_episodes, num_test_episodes, snapshot_interval, N=None):
   
    env = gym.make(env_name)
    E = num_episodes
    # Parameters
    
    # Initialize the result matrix D
    num_snapshots = E // snapshot_interval
    D = np.zeros((num_trials, num_snapshots))

    # Start the trials
    for trial in range(num_trials):
        print(f"Starting trial {trial + 1}/{num_trials}")
        # Initialize a new agent for each trial
        agent = PolicyGradient(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            lr_actor=1e-3,
            lr_critic=1e-3,
            mode=mode,
            n=128,
            gamma=0.99,
            device='cpu' ,
            **dict(max_steps = max_steps, N = N if mode == 'A2C' else None)
        )

        # Training loop
        total_episodes = 0
        for snapshot_idx in range(num_snapshots):
            # Train the agent for 'snapshot_interval' episodes
            print(f"  Training episodes {total_episodes + 1} to {total_episodes + snapshot_interval}")
            agent.run(
                env=env,
                max_steps=max_steps,
                num_episodes=snapshot_interval,
                train=True
            )
            total_episodes += snapshot_interval

            # Evaluate the agent by running 'num_test_episodes' episodes without training
            print(f"  Evaluating the policy at episode {total_episodes}")
            test_rewards = agent.run(
                env=env,
                max_steps=max_steps,
                num_episodes=num_test_episodes,
                train=False  # No training during evaluation
            )
            mean_return = np.mean(test_rewards)
            D[trial, snapshot_idx] = mean_return
            print(f"    Mean return over {num_test_episodes} test episodes: {mean_return}")

    # Close the environment
    env.close()
    return D

def parse_args():
    mode_choices = ['REINFORCE', 'BASELINE', 'A2C']

    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--mode', type=str, default='REINFORCE', choices=mode_choices, help='Mode to run the agent in')
    parser.add_argument('--n', type=int, default=64, help='The n to use for n step A2C')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=3500, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    parser.add_argument('--graph_name', type=str, default='REINFORCE', help='Graph Name')
    return parser.parse_args()



def main():
    args = parse_args()

    snapshot_interval = 100  # Freeze policy every 100 episodes
    num_test_episodes = 20  # Number of test episodes to run for evaluation
 
    D = run(args.env_name, args.max_steps, args.mode, args.num_runs, args.num_episodes, num_test_episodes, snapshot_interval, args.n)
    graph_agents(args.graph_name, D, args.max_steps, args.num_episodes)
   


if '__main__' == __name__:
    main()
