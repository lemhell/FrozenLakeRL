import gym
import numpy as np

ENV_NAME = "FrozenLake-v0"
env = gym.make(ENV_NAME)

try:
    Q_loaded = np.load("./" + ENV_NAME + ".npy")
    Q = Q_loaded
except FileNotFoundError as e:
    Q = np.zeros(shape=(env.observation_space.n, env.action_space.n))
    # Q = np.abs(np.random.randn(env.observation_space.n, env.action_space.n)/1000)

gamma = 0.999
learning_rate = 0.85
LAST_STATE = 15
episode_count = 50000


def get_max_q(state):
    if state == LAST_STATE:
        return 0
    max_q = 0
    for i in range(env.action_space.n):
        q_value = Q[state, i]
        if q_value > max_q:
            max_q = q_value
    return max_q


def get_next_action(state):
    row = Q[state, :]
    max_elems = np.where(row == np.amax(row))[0]
    if len(max_elems) == 1:
        return max_elems[0]
    else:
        return max_elems[np.random.randint(len(max_elems))]


env.render()
rewards = []
for i_episode in range(episode_count):
    observation = env.reset()
    r = 0
    for t in range(100):
        prev_obs = observation
        epsilon = 1/(i_episode + 1)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = get_next_action(observation)
        observation, reward, done, info = env.step(action)
        Q[prev_obs, action] += learning_rate*(reward + gamma*np.max(Q[observation, :]) - Q[prev_obs, action])
        r += reward
        if done:
            break
    rewards.append(r)

print(Q)
print(sum(rewards)/episode_count)

np.save("./" + ENV_NAME, Q)
