# Robustness of DRL


We compare standard DRL to our Decentralized Deep Reinforcement Learning approach with respect to robustness on hyperparameter variation.

Deep Decentralized Reinforcement Learning - locally structured architecture for DRL in a continuous locomotion control task on a four-legged simulated agent.

The repository holds results and implementation for training a decentralized control architecture of a four-legged robot.

For questions, please contact: Malte Schilling, mschilli@techfak.uni-bielefeld.de

## Abstract

Decentralization is a central characteristic of biological motor control that allows for fast responses relying on local sensory information. In contrast, the current trend of Deep Reinforcement Learning (DRL) based approaches to motor control follows a centralized paradigm using a single, holistic controller that has to untangle the whole input information space. This motivates to ask whether decentralization as seen in biological control architectures might also be beneficial for embodied sensori-motor control systems when using DRL. To answer this question, we provide an analysis and comparison of eight control architectures for adaptive locomotion that were derived for a four-legged agent, but with their *degree of decentralization* varying systematically between the extremes of fully centralized and fully decentralized. Our comparison shows that learning speed is significantly enhanced in distributed architectures—while still reaching the same high performance level of centralized architectures—due to smaller search spaces and *local costs* providing more focused information for learning. Second, we find an *increased robustness of the learning process* in the decentralized cases—it is less demanding to hyperparameter selection and less prone to becoming trapped in poor local minima. Finally, when examining *generalization to uneven terrains*—not used during training—we find best performance for an intermediate architecture that is decentralized, but integrates only *local information}*from both neighboring legs. Together, these findings demonstrate beneficial effects of distributing control into decentralized units and relying on local information. This appears as a promising approach towards more robust DRL and better generalization towards adaptive behavior.

![Visualization of our decentralized approach for locomotion of a simulated agent. In a) the modified quadruped walking agent is shown (derived from an OpenAI standard DRL task, but using more realistic dynamic parameters and a modified reward function). In b) a sketch of a (fully) decentralized approach is shown: On the one hand, control is handled concurrently and there are multiple controller (only two are shown in green in the visualization), one for each leg which reduces the action space of each individual controller (e.g., a_HR, t). On the other hand, only limited input information is used (gray arrows on the left, S_HR, t and S_FR, t), in this case only information from that particular leg which dramatically reduces the input state space. Control policies are trained using DRL which is driven by a reward signal (R_t, as a simplification this is shown as shared between all controller). This overall simplifies learning of robust behavior considerably and leads, as will be shown, to better generalization.](Results/Figures/Quantruped_Architecture.png)

## Overview Repository

The repository consists of multiple parts, all written in Python, and there are multiple experiments:

## Requirements

Code is written in python (version 3.6).

As a framework for DRL we used Ray (version 1.0.1) and RLlib. Ray is a framework that allows to efficiently distribute python tasks on clusters implementing an actor model for concurrent processing. It offers an easy and unified interface for running parallel tasks which are internally modeled as a graph of dependent tasks that can be executed. RLlib as part of Ray provides a scalable and efficient DRL framework which as well is centered around these actor-based computations that simplify setting up parallel computation and encapsulation of information.

Experiments were run in simulation using the Mujoco physics engine. We used Mujoco in version 1.50.1.68

Deep neural network models were realized in tensorflow 2 (version 2.3.1)

## Results

Summary of results: Shown are main results from the experiments for the two different control architectures.