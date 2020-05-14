import numpy as np
import sys
from collections import defaultdict
from gridworld_env import Gridworld
from gridworld_env import Action

NUM_EPISODES = 5_000
GAMMA = 0.95

def generate_episode_with_random_policy(env):
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = np.random.randint(0, env.action_space.n)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    N = defaultdict(lambda: np.zeros(env.action_space.n, dtype=int))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        episode = generate_episode(env)
        g = 0.0
        states, _, _ = zip(*episode)
        for step in reversed(range(0,len(episode))):
            (state, action, reward) = episode[step]
            g = reward + gamma * g
            # Is this the first visit? (First-Visit-MC)
            if not state in states[:step]:
                N[state][action] += 1
                Q[state][action] = Q[state][action] + (1/N[state][action])*(g - Q[state][action])
    print()
    return Q

env = Gridworld()
Q = mc_prediction_q(env, NUM_EPISODES, generate_episode_with_random_policy, GAMMA)

V = np.zeros(env.observation_space.n)
for k,v in Q.items():
    V[k] = np.round(np.max(v),0)

V = V.reshape((4, 4))
print("V-Table of gridworld with random policy:")
print(V)
