import numpy as np
import networkx as nx
import tensorflow as tf
import random
import gym


class DQN():

    def __init__(self, env, state_dim = 4, act_dim = 2):
        # Current Network:
        self._state_dim = state_dim
        self._act_dim = act_dim
        self._env = env

        self._state = tf.placeholder(dtype = tf.float32, shape = [None, state_dim])
        self._action = tf.placeholder(dtype = tf.float32, shape = [None, act_dim])

        self._W_s_1 = tf.get_variable(dtype = tf.float32, shape = [state_dim, 10], name = "W_s_1")
        self._W_a_1 = tf.get_variable(dtype = tf.float32, shape = [act_dim, 10], name = "W_a_1")
        self._b_1 = tf.get_variable(dtype = tf.float32, shape = [10], name = "b_1")

        self._h_1 = tf.nn.relu(
            tf.matmul(self._state, self._W_s_1) + tf.matmul(self._action, self._W_a_1) + self._b_1)
        self._W_2 = tf.get_variable(dtype=tf.float32, shape=[10, 1], name="W_2")
        self._b_2 = tf.get_variable(dtype=tf.float32, shape=[1], name="b_2")

        self._Q_hat = tf.reshape(tf.matmul(self._h_1, self._W_2) + self._b_2, shape = [-1, 1])

        # Target Network:

        self._W_s_1_target = tf.get_variable(dtype = tf.float32, shape = [state_dim, 10], name = "W_s_1_target")
        self._W_a_1_target = tf.get_variable(dtype = tf.float32, shape = [act_dim, 10], name = "W_a_1_target")
        self._b_1_target = tf.get_variable(dtype = tf.float32, shape = [10], name = "b_1_target")

        self._h_1_target = tf.nn.relu(
            tf.matmul(self._state, self._W_s_1_target) + tf.matmul(self._action,
                                                                        self._W_a_1_target) + self._b_1_target)
        self._W_2_target = tf.get_variable(dtype=tf.float32, shape=[10, 1], name="W_2_target")
        self._b_2_target = tf.get_variable(dtype=tf.float32, shape=[1], name="b_2_target")

        self._Q_hat_target = tf.reshape(tf.matmul(self._h_1_target, self._W_2_target) + self._b_2_target, shape = [-1, 1])

        self._weights = [self._W_s_1, self._W_a_1, self._b_1, self._W_2, self._b_2]
        self._weights_target = [self._W_s_1_target, self._W_a_1_target, self._b_1_target, self._W_2_target,
                                self._b_2_target]

    def train(
            self, n_epochs = 10, buffer_size = 1, discount = 1, tau = 0.999, lr = 0.1, seed = 0,
            save_every = 1, n_test_run = 10, weight_save_path = None, weight_load_path = None):
        self._saver = tf.train.Saver()
        self._replay_buffer = []
        self._y = tf.placeholder(dtype = tf.float32, shape = [None, 1])
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        # opt = tf.train.AdamOptimizer()
        # grad = opt.compute_gradients(self._Q_hat_target, var_list = [self._W_s, self._W_a, self._b])
        grad_W_s_1, grad_W_a_1, grad_b_1, grad_W_2, grad_b_2 = tf.gradients(ys = self._Q_hat, xs = self._weights)
        self._env.reset()
        self._env.seed(seed)

        survival_list = np.array([[0]])

        if weight_load_path is not None:
            self._saver.restore(self._sess, save_path = weight_load_path)
            print("Weights loaded successfully.")


        for epoch in range(n_epochs):
            print("Epoch: " + str(epoch))
            # Sample action and state
            state = self._env.reset()
            action_ind = self._env.action_space.sample()
            action = np.zeros(shape = self._act_dim)
            action[action_ind] = 1
            self.state = state
            new_state, reward, done, info = self._env.step(action_ind)

            self._replay_buffer.append((state, action, new_state, reward))
            # Pop some old sample:
            if len(self._replay_buffer) > buffer_size:
                ind = random.randrange(0, len(self._replay_buffer))
                self._replay_buffer.pop(ind)

            # Uniformly sample the replay buffer
            ind = random.randrange(0, len(self._replay_buffer))
            state, action, new_state, reward = self._replay_buffer[ind]

            # Compute the target:
            y = np.array([reward + discount * self.find_best_Q_target_v2(new_state)])
            y = y.reshape(1, 1)

            # Minimize the Bellman error:
            TD_error = self._Q_hat - self._y
            assign_W_s_1 = self._W_s_1.assign(self._W_s_1 - lr * grad_W_s_1 * TD_error)
            assign_W_a_1 = self._W_a_1.assign(self._W_a_1 - lr * grad_W_a_1 * TD_error)
            assign_b_1 = self._b_1.assign(self._b_1 - tf.reshape(lr * grad_b_1 * TD_error, [10]))
            assign_W_2 = self._W_2.assign(self._W_2 - lr * grad_W_2 * TD_error)
            assign_b_2 = self._b_2.assign(self._b_2 - tf.reshape(lr * grad_b_2 * TD_error, [1]))

            assigns = [assign_W_s_1, assign_W_a_1, assign_b_1, assign_W_2, assign_b_2]

            self._sess.run(assigns, feed_dict={self._state: [state],
                                               self._action: [action],
                                               self._y: y})

            # Update the target network using Polyak averaging:
            assign_W_s_1_target = self._W_s_1_target.assign(tau * self._W_s_1_target + (1 - tau) * self._W_s_1)
            assign_W_a_1_target = self._W_a_1_target.assign(tau * self._W_a_1_target + (1 - tau) * self._W_a_1)
            assign_b_1_target = self._b_1_target.assign(tau * self._b_1_target + (1 - tau) * self._b_1)
            assign_W_2_target = self._W_2_target.assign(tau * self._W_2_target + (1 - tau) * self._W_2)
            assign_b_2_target = self._b_2_target.assign(tau * self._b_2_target + (1 - tau) * self._b_2)
            assigns_target = [assign_W_s_1_target, assign_W_a_1_target, assign_b_1_target, assign_W_2_target,
                              assign_b_2_target]

            self._sess.run(assigns_target)

            test_list = np.array([self.test_run() for _ in range(n_test_run)])
            average_survival = np.mean(test_list)
            print("Test run: Survive for " + str(average_survival))
            survival_list = np.append(survival_list, np.array([average_survival]))

            if weight_save_path is not None and epoch % save_every == 0 and average_survival == np.max(survival_list):
                print("Average survival increases.")
                save_path = self._saver.save(self._sess, save_path = weight_save_path)
                print("Model's weights saved at %s" % save_path)
                if (average_survival >= 200):
                    print("Cartpole solved. Finish Training")
                    return

        print("Finish Training.")


    def find_best_Q_target(self, state):
        q_list = []
        for ind in range(self._act_dim):
            action = np.zeros(self._act_dim)
            action[ind] = 1
            q = self._sess.run(self._Q_hat_target, feed_dict = {self._state: [state], self._action: [action]})
            q_list.append(q)

        return max(q_list)

    def find_best_Q_target_v2(self, state):
        q_list = []
        for ind in range(self._act_dim):
            action = np.zeros(self._act_dim)
            action[ind] = 1
            q = self._sess.run(self._Q_hat, feed_dict = {self._state: [state], self._action: [action]})
            q_list.append(q)

        act_ind = np.argmax(q_list)
        action = np.zeros(self._act_dim)
        action[act_ind] = 1
        return self._sess.run(self._Q_hat_target, feed_dict = {self._state: [state], self._action: [action]})



    def act(self, state):
        q_list = []
        for ind in range(self._act_dim):
            action = np.zeros(self._act_dim)
            action[ind] = 1
            q = self._sess.run(self._Q_hat, feed_dict = {self._state: [state], self._action: [action]})
            q_list.append(q)

        return np.argmax(q_list)

    def test_run(self, timesteps = 1000):
        s = self._env.reset()
        for t in range(timesteps):
            action = self.act(s)
            s, reward, done, info = self._env.step(action)
            if done:
                return t





