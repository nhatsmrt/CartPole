import numpy as np
import networkx as nx
import tensorflow as tf
import random
import gym

class PolicyNetwork:

    def __init__(self, env, state_dim = 4, act_dim = 2, horizon = 200):
        self._env = env
        self._horizon = horizon
        self._act_dim = act_dim

        self._state = tf.placeholder(dtype = tf.float32, shape = [None, state_dim])
        self._W1 = tf.get_variable(dtype = tf.float32, shape = [state_dim, 10])
        self._b1 = tf.get_variable(dtype = tf.float32, shape = [10])
        self._fc1 = tf.nn.relu(tf.matmul(self._state, self._W1) + self._b1)

        self._W2 = tf.get_variable(dtype = tf.float32, shape = [10, act_dim])
        self._b2 = tf.get_variable(dtype = tf.float32, shape = [act_dim])
        self._z2 = tf.matmul(self._state, self._W2) + self._b2

        self._weights = [self._W1, self._b1, self._W2, self._b2]

        self._pi = tf.nn.softmax(self._z2, axis = -1)
        self._log_pi = tf.log(self._pi)


    def train(self, n_epochs, batch_size = 1, lr = 0.1):
        self._sess = tf.Session()

        for epoch in range(n_epochs):
            trajectory_batch_grad = []
            # Sample trajectories
            for batch in range(batch_size):
                state = self._env.reset()
                self._state = state
                sum_reward = 0
                trajectory_grad = []
                for t in range(self._horizon):
                    pi = self._sess.run(self._pi, feed_dict = {self._state: state})
                    # Sample the action index:
                    action = np.random.choice(self._act_dim, p = pi)
                    # Compute log grad:
                    list_grad = self._sess.run(
                        tf.gradients(
                            self._log_pi[action],
                            xs=self._weights),
                        feed_dict = {self._state: state})
                    trajectory_grad.append(list_grad)
                    state, reward, done, info = self._env.step(action)
                    sum_reward += reward

                    trajectory_grad = np.array(trajectory_grad)
                    trajectory_grad = np.sum(trajectory_grad, axis = 0) * sum_reward
                    trajectory_batch_grad.append(trajectory_grad)

                trajectory_batch_grad = np.array(trajectory_batch_grad)
                trajectory_batch_grad = np.sum(trajectory_batch_grad, axis = 0)
                assign_W1 = self._W1.assign(self._W1 + lr * trajectory_batch_grad[0])
                assign_b1 = self._b1.assign(self._b1 + lr * trajectory_batch_grad[1])
                assign_W2 = self._W2.assign(self._W2 + lr * trajectory_batch_grad[2])
                assign_b2 = self._b2.assign(self._b2 + lr * trajectory_batch_grad[3])
                assigns = [assign_W1, assign_b1, assign_W2, assign_b2]

                self._sess.run(assigns)

        print("Finish Training")

    def act(self, state, greedy = True):
        pi = self._sess.run(self._pi, feed_dict={self._state: state})
        if greedy:
            return np.argmax(pi, axis = -1)
        else:
            return np.random.choice(self._act_dim, p=pi)









