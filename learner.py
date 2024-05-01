import json
import time
import torch

import matplotlib.pyplot as plt

from environment_engine import EnvEngine, Action, Agent, CellType
from visualizer import Visualizer

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential, DistributionalDQNnet
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer, VDNMixer, MultiAgentConvNet
from torchrl.objectives import SoftUpdate, ValueEstimators
from torchrl.objectives.multiagent.qmixer import QMixerLoss

from torchsummary import summary



class TeamExplore():
    def __init__(self, n_agents=2, agent_abilities=[[1, 3], [1, 4]], seed=0) -> None:
        # Environment and agent parameters
        self.n_agents = n_agents
        self.agent_abilities = agent_abilities
        self.seed = seed

        torch.manual_seed(seed)

        self.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
        print(f"Using device: {self.device}")

        self.build_envs()

    # Build train and test environments
    def build_envs(self):
        self.env = EnvEngine(n_agents=self.n_agents, agent_abilities=[[1, 3], [1, 4]], map_size=32, device=self.device, seed=self.seed, max_steps=200)
        self.env_test = EnvEngine(n_agents=self.n_agents, agent_abilities=[[1, 3], [1, 4]], map_size=32, device=self.device, seed=self.seed, max_steps=200)

    # Build Q agent networks
    def build_q_agents(self):
        hidden_dim = 4096   # TODO: change this to match flattened output from cnn
        action_space = 5    # TODO: change this so it pulls from env.action_spec.space.n ?

        cnn = MultiAgentConvNet(
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            in_features=1,
            kernel_sizes=[5, 3, 3],
            num_cells=[32, 64, 64],
            strides=[2, 2, 1],
            paddings=[1, 1, 1],
            activation_class=torch.nn.ReLU,
            device=self.device
        )

        cnn_module = TensorDictModule(cnn, in_keys=[("agents", "observation")], out_keys=[("agents", "hidden")])

        mlp = MultiAgentMLP(
            n_agent_inputs=hidden_dim,
            n_agent_outputs=action_space,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            depth=2,
            num_cells=256,     # TODO: reduce this
            activation_class=nn.ReLU,
            device=self.device
        )

        mlp_module = TensorDictModule(mlp, in_keys=[("agents", "hidden")], out_keys=[("agents", "action_value")])

        # DistributionalDQNnet applies log softmax
        softmax = DistributionalDQNnet(in_keys=[("agents", "action_value")], out_keys=[("agents", "action_value")])

        value_module = QValueModule(
            action_value_key=("agents", "action_value"),
            out_keys=[
                ("agents", "action"),
                ("agents", "action_value"),
                ("agents", "chosen_action_value")
            ],
            spec=self.env.action_spec,
            action_space="one-hot"
        )

        # qnet = SafeSequential(cnn_module, mlp_module, value_module)
        qnet = SafeSequential(cnn_module, mlp_module, softmax, value_module)
        return qnet

    # Build Q agent networks
    def build_q_agents_shared_params(self):
        hidden_dim = 4096   # TODO: change this to match flattened output from cnn
        action_space = 5    # TODO: change this so it pulls from env.action_spec.space.n ?

        cnn = MultiAgentConvNet(
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            in_features=1,
            kernel_sizes=[5, 3, 3],
            num_cells=[32, 64, 64],
            strides=[2, 2, 1],
            paddings=[1, 1, 1],
            activation_class=torch.nn.ReLU,
            device=self.device
        )

        cnn_module = TensorDictModule(cnn, in_keys=[("agents", "observation")], out_keys=[("agents", "hidden")])

        mlp = MultiAgentMLP(
            n_agent_inputs=hidden_dim,
            n_agent_outputs=action_space,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            depth=2,
            num_cells=256,     # TODO: reduce this
            activation_class=nn.ReLU,
            device=self.device
        )

        mlp_module = TensorDictModule(mlp, in_keys=[("agents", "hidden")], out_keys=[("agents", "action_value")])

        # DistributionalDQNnet applies log softmax
        softmax = DistributionalDQNnet(in_keys=[("agents", "action_value")], out_keys=[("agents", "action_value")])

        value_module = QValueModule(
            action_value_key=("agents", "action_value"),
            out_keys=[
                ("agents", "action"),
                ("agents", "action_value"),
                ("agents", "chosen_action_value")
            ],
            spec=self.env.action_spec,
            action_space="one-hot"
        )

        # qnet = SafeSequential(cnn_module, mlp_module, value_module)
        qnet = SafeSequential(cnn_module, mlp_module, softmax, value_module)
        return qnet

    def train(self):
        # NOTE: epsilon updates every collector run, decreases linearly by < new_eps = old eps - ((eps_start - eps_end) / (eps_steps / frames_per_collector_run)) >
        # NOTE: learning rate optimizer steps every batch  < # steps total = ((frames_per_collector_run / batch_size) * n_epochs) * collector_runs >

        # Training params
        collector_runs = 20         # TODO: is this naming correct? should probably be something like "collections"
        frames_per_collector_run = 4096
        total_frames = frames_per_collector_run * collector_runs   
        memory_size = 100000         # TODO: increase this
        batch_size = 512             # TODO: big powers of 2
        gamma = 0.99
        tau = 0.005
        lr = 5e-5
        max_grad_norm = 40
        n_epochs = 10
        max_steps = 128     # Steps run during eval


        # # Fast training params
        # collector_runs = 5         # TODO: is this naming correct? should probably be something like "collections"
        # frames_per_collector_run = 128
        # total_frames = frames_per_collector_run * collector_runs
        # memory_size = 1000         # TODO: increase this
        # batch_size = 16             # TODO: big powers of 2
        # gamma = 0.99
        # tau = 0.005
        # lr = 5e-5
        # max_grad_norm = 40
        # n_epochs = 2
        # max_steps = 64     # Steps run during eval

        self.qnet = self.build_q_agents_shared_params()

        qnet_explore = TensorDictSequential(
            self.qnet,
            EGreedyModule(
                eps_init=1.0,
                eps_end=0.05,
                annealing_num_steps=50000,
                action_key=("agents", "action"),
                spec=self.env.action_spec
            )
        )

        mixer = TensorDictModule(
            module=QMixer(
                state_shape=self.env.observation_spec["state"].shape,
                mixing_embed_dim=32,
                n_agents=self.n_agents,
                device=self.device
            ),
            in_keys=[("agents", "chosen_action_value"), "state"],
            out_keys=["chosen_action_value"]
        )

        collector = SyncDataCollector(
            self.env,
            qnet_explore,
            device=self.device,
            storing_device=self.device,
            frames_per_batch=frames_per_collector_run,
            total_frames=total_frames
        )

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(memory_size, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size
        )

        loss_module = QMixerLoss(self.qnet, mixer, action_space="one-hot", delay_value=True)
        loss_module.set_keys(
            action_value=("agents", "action_value"),
            local_value=("agents", "chosen_action_value"),
            global_value="chosen_action_value",
            # action=env.action_key
            action=("agents", "action")
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
        
        target_net_updater = SoftUpdate(loss_module, eps=1-tau)

        optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

        episode_rewards = []

        total_frames = 0
        sampling_start = time.time()
        start_time = time.time()

        print("Enumerating collector")
        for i, tensordict_data in enumerate(collector):
            print(f"ITERATION: {i}")

            sampling_time = time.time() - sampling_start
            print("  sampling_time: {}".format(sampling_time))

            current_frames = tensordict_data.numel()
            total_frames += current_frames
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            training_tds = []
            training_start = time.time()
            for j in range(n_epochs):
                for k in range(frames_per_collector_run // batch_size):
                    subdata = replay_buffer.sample()

                    loss_vals = loss_module(subdata)

                    loss_value = loss_vals["loss"]

                    loss_value.backward()

                    training_tds.append(loss_vals.detach())

                    total_norm = torch.nn.utils.clip_grad_norm(
                        loss_module.parameters(), max_grad_norm
                    )
                    training_tds[-1].set("grad_norm", total_norm.mean())

                    optim.step()
                    optim.zero_grad()
                    target_net_updater.step()

            qnet_explore[1].step(frames=current_frames)
            collector.update_policy_weights_()

            training_time = time.time() - training_start
            print("  training_time: {}".format(training_time))

            iteration_time = sampling_time + training_time
            print("  iteration_time: {}".format(iteration_time))
            
            training_tds = torch.stack(training_tds)

            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                self.env_test.frames = []
                rollouts = self.env_test.rollout(
                    max_steps=max_steps,
                    policy=qnet_explore,
                    auto_cast_to_device=True,
                    break_when_any_done=False
                )

                # print("rewards: {}".format(rollouts["next", "episode_reward"][-1].item()))

                episode_rewards.append(rollouts["next", "episode_reward"][-1].item())

                self.save_actions_to_file("eval_{}_iter_{}.json".format(time.strftime("%Y%m%d-%H%M%S"), i), rollouts["agents", "action"], self.seed)

                evaluation_time = time.time() - evaluation_start

                print("  eval_time: {}".format(evaluation_time))

            sampling_start = time.time()

        print("total_time: {}".format(time.time() - start_time))

        # Visualize last eval rollout
        if 1:
            vis = Visualizer()
            vis.init_game_vis()
            self.env_test.reset()
            vis.visualize_action_set(self.env_test, rollouts["agents", "action"])

        plt.plot(episode_rewards)
        plt.show()

    # Save tensor of actions to json file
    def save_actions_to_file(self, fname, actions, seed):
        # for step in actions:
        #     print("step: {}".format(step))
        list_actions = [step.tolist() for step in actions]
        dict_actions = {"seed" : seed, "actions" : list_actions}

        with open(fname, 'w') as fp:
            json.dump(dict_actions, fp)




if __name__ == "__main__":
    te = TeamExplore()
    te.train()
