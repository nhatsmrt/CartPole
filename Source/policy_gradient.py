import numpy as np
import networkx as nx
import tensorflow as tf
import random
import gym

class PolicyNetwork:

    def __init__(self, env, state_dim = 4, act_dim = 2, horizon = 500, discount = 0.95):
        self._env = env
        self._horizon = horizon
        self._act_dim = act_dim
        self._state_dim = state_dim
        self._discount = discount


        self._state = tf.placeholder(dtype = tf.float32, shape = [None, state_dim])
        self._W1 = tf.get_variable(dtype = tf.float32, shape = [state_dim, 10], name = "W1")
        self._b1 = tf.get_variable(dtype = tf.float32, shape = [10], name = "b1")
        self._fc1 = tf.nn.relu(tf.matmul(self._state, self._W1) + self._b1)

        self._W2 = tf.get_variable(dtype = tf.float32, shape = [10, act_dim], name = "W2")
        self._b2 = tf.get_variable(dtype = tf.float32, shape = [act_dim], name = "b2")
        self._z2 = tf.matmul(self._fc1, self._W2) + self._b2

        self._weights = [self._W1, self._b1, self._W2, self._b2]

        self._pi = tf.nn.softmax(self._z2, axis = -1)
        self._log_pi = tf.log(self._pi)


    def train(self, n_epochs, batch_size = 10, lr = 0.1, seed = 0, weight_save_path = None, weight_load_path = None):
        self._saver = tf.train.Saver()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._env.reset()
        self._env.seed(seed)
        list_average_survival = []

        if weight_load_path is not None:
            self._saver.restore(self._sess, save_path = weight_load_path)
            print("Weights loaded successfully.")

        for epoch in range(n_epochs):
            print("Epoch " + str(epoch))
            trajectory_batch_grad = []
            batch_cum_sum_reward = []
            survival_list = []
            # Sample trajectories
            for batch in range(batch_size):
                state = self._env.reset()
                # print(state)
                # sum_reward = 0
                cum_sum_reward = np.zeros(shape = (1, 1))
                trajectory_grad = []

                done = False
                survival_counter = 0
                discount = 1
                for t in range(self._horizon):
                    if not done:
                        # print([state])
                        pi = self._sess.run(
                            self._pi,
                            feed_dict = {self._state: [state]})
                        # Sample the action index:
                        # print(pi)
                        action = np.random.choice(self._act_dim, p = pi[0])
                        # Compute log grad:
                        list_grad = self._sess.run(
                            tf.gradients(
                                self._log_pi[:, action],
                                xs=self._weights),
                            feed_dict = {self._state: [state]})
                        trajectory_grad.append(list_grad)
                        state, reward, done, info = self._env.step(action)
                        cum_sum_reward += reward * discount
                        discount *= self._discount
                        if t < self._horizon - 1 and not done:
                            cum_sum_reward = np.append(cum_sum_reward, [[0]], axis = 0)
                        survival_counter += 1
                    # else:
                    #     list_grad = [
                    #         np.zeros(shape = (self._state_dim, 10)),
                    #         np.zeros(shape=(10)),
                    #         np.zeros(shape=(10, self._act_dim)),
                    #         np.zeros(shape=(self._act_dim))
                    #     ]
                    #     trajectory_grad.append(list_grad)
                    #     if t < self._horizon - 1:
                    #         cum_sum_reward = np.append(cum_sum_reward, [[0]], axis = 0)
                survival_list.append(survival_counter)
                print("survival " + str(survival_counter))

                cum_sum_reward = (cum_sum_reward - np.mean(cum_sum_reward)) / np.std(cum_sum_reward)
                batch_cum_sum_reward.append(cum_sum_reward)
                trajectory_grad = np.array(trajectory_grad)
                # print(trajectory_grad)
                # trajectory_grad = np.sum(trajectory_grad, axis = 0)
                trajectory_batch_grad.append(trajectory_grad)

            mean_survive = np.mean(survival_list)
            print("Average survival " + str(mean_survive))
            list_average_survival.append(mean_survive)
            if mean_survive == max(list_average_survival):
                print("Average survival increases")
                if weight_save_path is not None:
                    save_path = self._saver.save(self._sess, save_path=weight_save_path)
                    print("Model's weights saved at %s" % save_path)

            if mean_survive > 200:
                print("CartPole solved.")
                return

            trajectory_batch_grad = np.array(trajectory_batch_grad)
            batch_cum_sum_reward = np.array(batch_cum_sum_reward)

            # Computing the baseline:
            numerator = np.sum(
                self.array_map(lambda x: np.sum(x, axis = 0), trajectory_batch_grad * trajectory_batch_grad * batch_cum_sum_reward),
                axis = 0)
            denominator = np.sum(
                self.array_map(lambda x: np.sum(x, axis = 0), trajectory_batch_grad * trajectory_batch_grad),
                axis = 0)
            baseline = numerator / (denominator + 1.e-8)
            batch_cum_sum_reward = self.array_map(lambda x: x - baseline[None, :], batch_cum_sum_reward)
            trajectory_batch_grad = trajectory_batch_grad * batch_cum_sum_reward
            trajectory_batch_grad = self.array_map(lambda x: np.sum(x, axis = 0), trajectory_batch_grad)

            trajectory_batch_grad = np.mean(trajectory_batch_grad, axis = 0)
            # print(trajectory_batch_grad)
            assign_W1 = self._W1.assign(self._W1 + lr * trajectory_batch_grad[0])
            assign_b1 = self._b1.assign(self._b1 + lr * trajectory_batch_grad[1])
            assign_W2 = self._W2.assign(self._W2 + lr * trajectory_batch_grad[2])
            assign_b2 = self._b2.assign(self._b2 + lr * trajectory_batch_grad[3])
            assigns = [assign_W1, assign_b1, assign_W2, assign_b2]

            self._sess.run(assigns)

            # weight_check = self._sess.run(self._weights)

                    # print(weight_check)

        print("Finish Training")


    def act(self, state, greedy = True):
        pi = self._sess.run(self._pi, feed_dict={self._state: [state]})
        if greedy:
            return np.argmax(pi, axis = -1)
        else:
            return np.random.choice(self._act_dim, p = pi[0])

    def array_map(self, f, array):
        return np.array(list(map(f, array)))









