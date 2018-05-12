import tensorflow as tf
import numpy as np
import gym


class QNetwork:

    def __init__(self, state_dim = 1, action_dim = 2):
        self._state_dim = state_dim
        self._action_dim = action_dim

        self._state = tf.placeholder(shape = [None, state_dim], dtype = tf.float32)
        self._W =  tf.get_variable(shape = [state_dim, action_dim], name = "W", dtype = tf.float32)
        self._b = tf.get_variable(shape = [action_dim])

        self._q_op = tf.matmul(self._state, self._W) + self._b
        self._prediction = tf.argmax(self._q_op, axis = -1)


    def fit(self, env, n_epochs = 5, e = 0.1):
        self._q_true = tf.placeholder(shape = [None, self._action_dim])
        self._loss = tf.reduce_sum(tf.square(self._q_op - self._q_true))

        self._optimizer = tf.train.AdamOptimizer()
        train_step = self._optimizer.minimize(self._loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        for _ in range(n_epochs):
            # Reset the environment and get new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            while j < 99:
                j += 1
                # self._sess.run(train_step, feed_dict = {self._state: state, self._q_true: q_true})
                # Choose an action greedily (with a probability of e of choosing random) from the Q-Network:
                a, all_q = self._sess.run([self._prediction, self._W], feed_dict = {self._state: np.identity(self._state_dim)[s : s+1]})
                if np.random.rand(1) < e:
                    a[0] = env.action.space.sample()

                # Get new state and reward from environment:
                s11, r =  env.step(a[0])

                # Feed the new state to the Q-Network to obtain the Q values:
                Q1 = self._sess.run(self._q_op, feed_dict = {self._state: s1})







