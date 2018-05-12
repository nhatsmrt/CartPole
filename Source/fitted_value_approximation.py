import tensorflow as tf
import numpy as np
import gym
import copy


class FittedValueApproximator:

    def __init__(self, env, state_dim = 4, action_dim = 2, discount_factor = 1):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._env = copy.deepcopy(env)
        self._discount_factor = discount_factor

        self._state = tf.placeholder(shape = [None, state_dim], dtype = tf.float64)
        self._W =  tf.get_variable(name = "W", dtype = tf.float64, initializer = tf.zeros(shape = [state_dim, 1], dtype = tf.float64))
        self._b = tf.get_variable(name = "b", initializer = tf.zeros(shape = [1], dtype = tf.float64))

        self._value_approx = tf.matmul(self._state, self._W) + self._b


    def train(self, n_epochs = 5, e = 0.1, m = 10, k = 1, seed = 0):
        self._y = tf.placeholder(shape = [None, 1], dtype = tf.float64)
        self._loss = tf.reduce_mean(0.5 * tf.square(self._value_approx - self._y))

        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._env.seed(seed)
        sample_states = []
        for i in range(m):
            sample_states.append(self._env.reset())


        for epoch in range(n_epochs):
            all_y = []
            print("Training epoch " + str(epoch) + ":")
            for i in range(m):
                # print("Sample " + str(i))
                all_q = []
                self._env.state = sample_states[i]
                for a in range(2):
                    states_list = np.array([])
                    rewards_list = np.array([])
                    values_list = np.array([])
                    for j in range(k):
                        state, reward, done, info = self._env.step(a)
                        # states_list = np.insert(states_list, [0], [state])
                        # rewards_list = np.insert(rewards_list, - 1, reward)
                        value_approx = self._sess.run(self._value_approx, feed_dict = {self._state: np.array([state])})
                        # print(value_approx)
                        values_list = np.insert(values_list, [0], [value_approx])
                    q = 1 + self._discount_factor * np.mean(values_list)
                    all_q.append(q)
                # print(all_q)
                all_y.append(np.max(all_q))

            self._sess.run(self._train_step, feed_dict = {self._state: np.array(sample_states), self._y: np.array(all_y).reshape(-1, 1)})
            W = self._sess.run(self._W)
            # print(W)

    def act(self, state_original, k = 1):
        all_q = []
        for a in range(2):
            states_list = np.array([])
            rewards_list = np.array([])
            values_list = np.array([])
            for j in range(k):
                self._env.state = state_original
                state, reward, done, info = self._env.step(a)
                # states_list = np.insert(states_list, [0], [state])
                # rewards_list = np.insert(rewards_list, - 1, reward)
                value_approx = self._sess.run(self._value_approx, feed_dict={self._state: np.array([state])})
                values_list = np.insert(values_list, [0], [value_approx])
            q = 1 + self._discount_factor * np.sum(values_list) / k
            all_q.append(q)

        return np.argmax(all_q)







