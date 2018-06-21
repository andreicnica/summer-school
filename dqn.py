
# coding: utf-8

# In[1]:


# #### Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

#
# Reinforcement Learning (DQN) tutorial
# =====================================
# **Author**: `Adam Paszke <https://github.com/apaszke>`_
#

# In[2]:


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


#  Gathering environment

from gathering_mae.single_agent_wrapper import SingleAgentGatheringEnv
from configs.utils import load_config

# Get default config
cfg = load_config("configs/default_env.yaml")
cfg.base_map = "configs/maps/map32_4rooms_static.txt"
env = SingleAgentGatheringEnv(cfg)
obs_size: torch.Size = env.observation_space
no_actions = 4


# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#
#
#

# In[4]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Now, let's define our model. But first, let quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# $R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t$, where
# $R_{t_0}$ is also known as the *return*. The discount,
# $\gamma$, should be a constant between $0$ and $1$
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
#
# The main idea behind Q-learning is that if we had a function
# $Q^*: State \times Action \rightarrow \mathbb{R}$, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# \begin{align}\pi^*(s) = \arg\!\max_a \ Q^*(s, a)\end{align}
#
# However, we don't know everything about the world, so we don't have
# access to $Q^*$. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# $Q^*$.
#
# For our training update rule, we'll use a fact that every $Q$
# function for some policy obeys the Bellman equation:
#
# \begin{align}Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))\end{align}
#
# The difference between the two sides of the equality is known as the
# temporal difference error, $\delta$:
#
# \begin{align}\delta = Q(s, a) - (r + \gamma \max_a Q(s', a))\end{align}
#
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of $Q$ are very noisy. We calculate
# this over a batch of transitions, $B$, sampled from the replay
# memory:
#
# \begin{align}\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)\end{align}
#
# \begin{align}\text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}\end{align}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing $Q(s, \mathrm{left})$ and
# $Q(s, \mathrm{right})$ (where $s$ is the input to the
# network). In effect, the network is trying to predict the *quality* of
# taking each action given the current input.
#
#
#

# In[5]:


class DQN(nn.Module):

    def __init__(self, in_size: torch.Size, out_size: torch.Size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_size[0], 16, kernel_size=3, stride=1)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(288, out_size[0])

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_size: torch.Size, out_size: torch.Size):
        super(MLP, self).__init__()
        in_units = reduce(mul, in_size, 1)
        hidden_size = 256
        self.ln1 = nn.Linear(in_units, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.ln2 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.Linear(hidden_size, hidden_size)
        # self.ln4 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        self.head = nn.Linear(hidden_size, out_size[0])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu((self.ln1(x)))
        x = F.relu((self.ln2(x)))
        x = F.relu((self.ln3(x)))
        # x = F.relu(self.ln1(x))
        # x = F.relu(self.ln2(x))
        # x = F.relu(self.ln3(x))
        # x = F.relu(self.ln4(x))
        return self.head(x)


# Input extraction
# ^^^^^^^^^^^^^^^^
#
# The code below are utilities for extracting and processing rendered
# images from the environment. It uses the ``torchvision`` package, which
# makes it easy to compose image transforms. Once you run the cell it will
# display an example patch that it extracted.
#
#
#

# In[6]:


# Vizualize env capture
def convert_obs_to_screen(obs):
    obs = obs.clone()
    obs.mul_(255).int()
    obs = obs.numpy().astype(np.uint8)
    return obs


obs = env.reset()
screen = env.render()
view_obs = convert_obs_to_screen(obs)

# Plot full map + partial
plt.figure()
plt.imshow(screen,
           interpolation='none')
plt.title('Example view of game full map & agent observation')
plt.show()

# Plot actual observation
plt.figure()
plt.imshow(view_obs,
           interpolation='none')
plt.title('Example view of agent actual received observation')
plt.show()


# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#
#
#

# In[7]:


BATCH_SIZE = 128
GAMMA = 0.4
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TARGET_UPDATE = 4
OPTIMIZE_FREQ = 5

policy_net = DQN(in_size=obs_size, out_size=torch.Size([no_actions])).to(device)
target_net = DQN(in_size=obs_size, out_size=torch.Size([no_actions])).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
memory = ReplayMemory(1000)


steps_done = 0


def select_action(state, eval=False):
    global steps_done
    act = None
    out = None

    if not eval:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if steps_done % (EPS_DECAY // 4) == 0:
            print(f"Crt eps: {eps_threshold}")

        if sample < eps_threshold:
            act = torch.tensor([[random.randrange(no_actions)]], device=device, dtype=torch.long)

    policy_net.eval()
    with torch.no_grad():
        out = policy_net(state)
        act = out.max(1)[1].view(1, 1)
    return act, out


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes $Q(s_t, a_t)$ and
# $V(s_{t+1}) = \max_a Q(s_{t+1}, a)$, and combines them into our
# loss. By defition we set $V(s) = 0$ if $s$ is a terminal
# state. We also use a target network to compute $V(s_{t+1})$ for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#
#
#

# In[8]:
def process_obs(obs):
    obs = obs.unsqueeze(0).unsqueeze(0)
    obs.add_(-0.6588499999999999).mul_(2.)
    return obs


def optimize_model():
    policy_net.train()
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8).to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-0.25, 0.25)
    optimizer.step()


# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.
#
#
#

# In[ ]:


eval_episodes = 5

def eval_agent(ep_no):
    eval_returns = []
    eval_lengths = []
    for _ in range(eval_episodes):
        ep_return = 0
        state = process_obs(env.reset())

        for t in count():
            action, out = select_action(state.to(device), eval=True)
            action = action.cpu()
            screen, reward, done, _ = env.step(action.item())
            state = process_obs(screen)

            ep_return += reward
            if done:
                eval_returns.append(ep_return)
                eval_lengths.append(t + 1)
                ep_return = 0
                break
        # print(out.cpu().numpy())
        # print(EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))

    print(f"[Eval  {ep_no:d}] Return:", np.mean(eval_returns), " | Ep. length:", np.mean(eval_lengths))


num_episodes = 5000
returns = []
all_steps = 0
for i_episode in range(num_episodes):
    ep_return = 0
    # Initialize the environment and state

    state = process_obs(env.reset())

    for t in count():
        # Select and perform an action
        action, _ = select_action(state.to(device))
        action = action.cpu()
        current_screen, reward, done, _ = env.step(action.item())
        ep_return += reward
        current_screen = process_obs(current_screen)
        reward = torch.tensor([reward])
        all_steps += 1

        # Observe new state
        if not done:
            next_state = current_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward.cpu())

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        if i_episode > 10 and all_steps % OPTIMIZE_FREQ == 0:
            optimize_model()

        if done:
            returns.append(ep_return)
            episode_durations.append(t + 1)
            ep_return = 0
            if i_episode % 1 == 0:
                print(f"[Train {i_episode:d}] Return:", np.mean(returns),
                      " | Ep. length:", np.mean(episode_durations))
                episode_durations.clear()
                returns = returns[-eval_episodes:]
                eval_agent(i_episode)
            # plot_durations()

            # if i_episode % 100 == 99:
            #     eval_env_visual(env, policy_net)
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
