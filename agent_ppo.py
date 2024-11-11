import copy
import json
import time
import torch

import matplotlib.pyplot as plt

from pathlib import Path

from torchrl.data.tensor_specs import TensorSpec

from environment_engine import EnvEngine
from visualizer import Visualizer

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
# from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential, DistributionalDQNnet
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer, MultiAgentConvNet
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss

class AgentPPO():
    def __init__(self, n_agents=2, agent_abilities=[[1,3], [1,4]], seed=0):
        # Environment and agent parameters
        self.n_agents = n_agents
        self.agent_abilities = agent_abilities
        self.seed = seed

        torch.manual_seed(seed)

        self.device = "cpu" if not torch.cuda.device_count() else "cuda"
        print(f"Using device: {self.device}")

    