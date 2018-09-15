# CartPole
## Introduction
This project is based on OpenAI's cartpole environment.
The agent's task is to move the cart pole left and right while balancing an inverted pendulum on top of it.
<br/>
NOTE: This project is writen in Tensorflow 1.9.
## Approach
A neural network is used to approximate the policy (i.e the distribution of actions given states). This neural network is then trained using policy gradient (REINFORCE algorithm).
