import pickle
import sys
import gym
import tqdm

from agent import Agent


def learn(env, n_episodes, alpha, gamma, epsilon=1, eps_decay=0.999, eps_min=0.05):
    agent = Agent(env)
    state = None

    for i in tqdm.tqdm(range(1, n_episodes + 1)):
        env.reset()
        # Decay epsilon
        epsilon = max(epsilon * eps_decay, eps_min)
        # Do one episode
        episode = []
        while True:
            action = agent.select_action(env, state=state, epsilon=epsilon)
            state, reward, done, _, _ = env.step(action)

            episode.append((state, action, reward))
            # agent.episodes.append((state, action, reward))
            if done:
                break

        # Update Q table using the epsilon-greedy-policy
        agent.learn(episode, alpha, gamma)
    agent.get_best_policy()

    return agent


if __name__ == "__main__":
    env = gym.make('Blackjack-v1', render_mode="rgb_array")
    agent = learn(env, 1_500_000, 0.02, gamma=1.0, epsilon=1, eps_decay=0.999, eps_min=0.05)
    with open('optimal_policy.pkl', 'wb') as handle:
        pickle.dump(agent.policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
