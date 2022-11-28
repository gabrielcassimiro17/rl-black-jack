import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, env):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.episodes = []
        self.policy = None

    def select_action(self, env, state, epsilon):
        Q = self.q_table
        if epsilon > np.random.random():
            return env.action_space.sample()
        elif state in Q:
            return np.argmax(Q[state])
        return env.action_space.sample()

    def learn(self, episode, alpha, gamma):
        Q = self.q_table
        G = 0

        # Save all states, actions and rewards in different variables
        for state, action, reward in episode[::-1]:
            G += reward
            Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (G)

        self.q_table = Q

    def get_best_policy(self):
        self.policy = dict((k, np.argmax(v)) for k, v in self.q_table.items())
