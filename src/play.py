import pickle
import gym
from tqdm import tqdm


def play_game(n_games, policy):
    env = gym.make('Blackjack-v1', render_mode="rgb_array")
    won = 0
    for i in tqdm(range(n_games)):
        state, _ = env.reset()
        while True:
            action = policy[state]
            state, reward, done, _, _ = env.step(action)
            if done:
                if reward == 1:
                    won += 1
                break
    print(won / n_games)


if __name__ == '__main__':
    with open('optimal_policy.pkl', 'rb') as handle:
        policy = pickle.load(handle)
    play_game(500_000, policy)
