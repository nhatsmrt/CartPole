import tensorflow as tf
import numpy as np
import gym
import copy


class SimpleQNetwork:

    def __init__(self, env, state_dim = 4, action_dim = 2, discount_factor = 1 ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._env = copy.deepcopy(env)
        self._discount_factor = discount_factor

        self._state = tf.placeholder(shape = [None, state_dim], dtype = tf.float32)
        self._action = tf.placeholder(shape = [None, action_dim], dtype = tf.float32)
        self._W_s =  tf.get_variable(shape = [state_dim, 1], name = "W_s", dtype = tf.float32)
        self._W_a =  tf.get_variable(shape = [action_dim, 1], name = "W_a", dtype = tf.float32)
        self._b = tf.get_variable(shape = [1], name = "b")

        self._q_approx = tf.matmul(self._state, self._W_s) + tf.matmul(self._state, self._W_s) + self._b


    def train(self, n_epochs = 5, e = 0.1, m = 10, k = 1, seed = 0):
        self._y = tf.placeholder(shape = [1], dtype = tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._y - self._q_approx))

        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._env.seed(seed)

        for epoch in range(n_epochs):
            all_y = []
            print("Training epoch " + str(epoch) + ":")
            sample_states = []
            sample_actions = []
            for i in range(m):
                # print("Sample " + str(i))
                sample_states.append(self._env.reset())
                sample_action = self._env.action_space.sample()
                sample_actions.append(self.action_encode(sample_action))
                all_q = []
                self._env.state = sample_states[i]
                self._env.step(sample_action)
                # Compute y:
                # Compute \mathcal{E}} [r + \gamma \max_{a'} Q(s',a'|\theta_{i - 1})|s,a]
                    # Compute
                for j in range(k):
                    q_list = np.array([])
                    # Compute \max_{a'} Q(s',a'|\theta_{i - 1})
                    for a in range(2):
                        action = self.action_encode(a)
                        state, reward, done, info = self._env.step(a)
                        # states_list = np.insert(states_list, [0], [state])
                        # rewards_list = np.insert(rewards_list, - 1, reward)
                        q_approx = self._sess.run(self._q_approx, feed_dict = {self._state: np.array([state]), self._action: action})
                        # print(value_approx)
                        q_list = np.insert(q_list, [0], [q_approx])
                    q = np.max(q_list)
                    all_q.append(q)
                # Compute y = \mathcal{E}} [r + \gamma \max_{a'} Q(s',a'|\theta_{i - 1})|s,a] = 1 + gamma * self._discount_factor * np.mean(all_q)
                all_y.append(1 + self._discount_factor * np.mean(all_q))

            self._sess.run(self._train_step, feed_dict = {self._state: np.array(sample_states),
                                                          self._action: sample_actions,
                                                          self._y: np.array(all_y).reshape(-1, 1)})


    def act(self, state_original, k = 1):
        q_list = np.array([])
        # Compute a = \argmax_{a'} Q(s,a'|\theta)
        for a in range(2):
            action = self.action_encode(a)
            q_approx = self._sess.run(self._q_approx, feed_dict={self._state: np.array([state_original]), self._action: action})
            q_list = np.insert(q_list, [0], [q_approx])

        return np.argmax(q_list)

    def action_encode(self, action):
        return np.eye(N = self._action_dim)[action]




# class QNetwork:
#
#     def __init__(self, state_dim = 1, action_dim = 2):
#         self._state_dim = state_dim
#         self._action_dim = action_dim
#
#         self._state = tf.placeholder(shape = [None, state_dim], dtype = tf.float32)
#         self._W =  tf.get_variable(shape = [state_dim, action_dim], name = "W", dtype = tf.float32)
#         self._b = tf.get_variable(shape = [action_dim])
#
#         self._q_op = tf.matmul(self._state, self._W) + self._b
#         self._prediction = tf.argmax(self._q_op, axis = -1)
#
#
#     def fit(self, env, n_epochs = 5, e = 0.1):
#         self._q_true = tf.placeholder(shape = [None, self._action_dim])
#         self._loss = tf.reduce_sum(tf.square(self._q_op - self._q_true))
#
#         self._optimizer = tf.train.AdamOptimizer()
#         train_step = self._optimizer.minimize(self._loss)
#
#         self._sess = tf.Session()
#         self._sess.run(tf.global_variables_initializer())
#
#         for _ in range(n_epochs):
#             # Reset the environment and get new observation
#             s = env.reset()
#             rAll = 0
#             d = False
#             j = 0
#             while j < 99:
#                 j += 1
#                 # self._sess.run(train_step, feed_dict = {self._state: state, self._q_true: q_true})
#                 # Choose an action greedily (with a probability of e of choosing random) from the Q-Network:
#                 a, all_q = self._sess.run([self._prediction, self._W], feed_dict = {self._state: np.identity(self._state_dim)[s : s+1]})
#                 if np.random.rand(1) < e:
#                     a[0] = env.action.space.sample()
#
#                 # Get new state and reward from environment:
#                 s11, r =  env.step(a[0])
#
#                 # Feed the new state to the Q-Network to obtain the Q values:
#                 Q1 = self._sess.run(self._q_op, feed_dict = {self._state: s1})







