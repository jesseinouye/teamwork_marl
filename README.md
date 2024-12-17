# teamwork_marl
Teamwork-based map exploration through multi-agent reinforcement learning (MARL).

This project implements the PPO and QMIX learning algorithms, training multiple agents with heterogeneous capabilities to efficiently and quickly explore a randomly generated map.

**Example:**

Two agents explore a random map. The red agent can move through green tiles while the purple agent can move through blue tiles. The left side of the gif below shows the "ground truth" global state, while the right side shows the red agent's observation combined with the purple agent's observation history (i.e. the shared local observations).

![til](media/example_run.gif)

## Project Breakdown

### Map Engine

The Map Engine randomly generates explorable areas and handles agent actions. It receives agent action tensors as input and outputs the global ground truth state and each agent's shared local observations.

### PPO Implementation

Multi-agent PPO (MAPPO) is implemented with PyTorch / TorchRL. Each agent (actor) receives their local observation and the shared global map observations (e.g. ground state minus locations of other agents). A centralized critic model receives the global state (observations and locations of every agent).

The agent network structure is: 3 layer CNN -> 2 layer MLP -> policy

### QMIX Implementation

The QMIX algorithm is implemented with PyTorch / TorchRL. Each agent receives their local observation and the shared global map observations (e.g. ground state minus locations of other agents). A centralized Q mixer receives the global state (observations and locations of every agent).

The agent network structure is: 3 layer CNN -> 2 layer MLP -> value module -> q values

## Future work

- Different actor networks:
  - I'd like to try this with different network architectures using some form of memory (RNN, transformer, etc.). Not sure exactly what this would look like.
- Speed up environment:
  - Right now training is really slow (~300 steps per second). I don't know how much I could really speed it up in python without something like multiprocessing. Would be cool to try building it in C/C++ with some framework (Madrona?)... 


## Versions

TODO: get versions of python, packages, cuda, etc.