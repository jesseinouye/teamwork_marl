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


# class CustomEGreedyModule(EGreedyModule):
#     def __init__(self, spec: TensorSpec, eps_init: float = 1, eps_end: float = 0.1, annealing_num_steps: int = 1000, *, action_key: str | torch.Tuple[str] | None = "action", action_mask_key: str | torch.Tuple[str] | None = None):
#         super().__init__(spec, eps_init, eps_end, annealing_num_steps, action_key=action_key, action_mask_key=action_mask_key)

#     def step(self, frames: int = 1) -> None:
#         """A step of epsilon decay.

#         After `self.annealing_num_steps` calls to this method, calls result in no-op.

#         Args:
#             frames (int, optional): number of frames since last step. Defaults to ``1``.

#         """
#         if self.eps == self.eps_end:
#             self.eps.data[0] = self.eps_init.item()
#         else:
#             for _ in range(frames):
#                 self.eps.data[0] = max(
#                     self.eps_end.item(),
#                     (
#                         self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
#                     ).item(),
#                 )



class TeamExplore():
    def __init__(self, n_agents=2, agent_abilities=[[1, 3], [1, 4]], seed=0) -> None:
        # Environment and agent parameters
        self.n_agents = n_agents
        self.agent_abilities = agent_abilities
        self.seed = seed

        torch.manual_seed(seed)

        self.device = "cpu" if not torch.cuda.device_count() else "cuda"
        # self.device = "cpu"
        print(f"Using device: {self.device}")

        self.build_envs()

    # Build train and test environments
    def build_envs(self):
        self.env = EnvEngine(n_agents=self.n_agents, agent_abilities=self.agent_abilities, map_size=32, device=self.device, seed=self.seed, max_steps=512)
        self.env_test = EnvEngine(n_agents=self.n_agents, agent_abilities=self.agent_abilities, map_size=32, device=self.device, seed=self.seed, max_steps=512)

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

        qnet = SafeSequential(cnn_module, mlp_module, value_module)
        # qnet = SafeSequential(cnn_module, mlp_module, softmax, value_module)
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
        # softmax = DistributionalDQNnet(in_keys=[("agents", "action_value")], out_keys=[("agents", "action_value")])

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

        qnet = SafeSequential(cnn_module, mlp_module, value_module)
        # qnet = SafeSequential(cnn_module, mlp_module, softmax, value_module)
        return qnet

    def train(self):
        # NOTE: epsilon updates every collector run, decreases linearly by : new_eps = old eps - ((eps_start - eps_end) / (eps_steps / frames_per_collector_run))
        # NOTE: learning rate optimizer steps every batch : # steps total = ((frames_per_collector_run / batch_size) * n_epochs) * collector_runs

        # Main training params
        collector_runs = 300
        frames_per_collector_run = 8192
        total_frames = frames_per_collector_run * collector_runs   
        memory_size = 100000
        batch_size = 512
        eps_init = 1.0
        eps_end = 0.05
        eps_num_steps = 100000
        gamma = 0.99
        tau = 0.005
        lr = 5e-5
        max_grad_norm = 30
        n_epochs = 5
        max_steps = 300     # Steps run during eval

        # Tmp training params
        collector_runs = 5
        frames_per_collector_run = 8192
        total_frames = frames_per_collector_run * collector_runs   
        memory_size = 50000
        batch_size = 512
        eps_init = 1.0
        eps_end = 0.05
        eps_num_steps = 100000
        gamma = 0.99
        tau = 0.005
        lr = 5e-5
        max_grad_norm = 30
        n_epochs = 5
        max_steps = 300     # Steps run during eval

        # # Test training params
        # collector_runs = 3000
        # frames_per_collector_run = 2048
        # total_frames = frames_per_collector_run * collector_runs   
        # memory_size = 100000
        # batch_size = 512
        # eps_init = 1.0
        # eps_end = 0.05
        # eps_num_steps = 100000
        # gamma = 0.99
        # tau = 0.005
        # lr = 5e-4
        # max_grad_norm = 40
        # n_epochs = 10
        # max_steps = 300     # Steps run during eval


        # # Fast training params
        # collector_runs = 1
        # frames_per_collector_run = 1
        # total_frames = frames_per_collector_run * collector_runs
        # memory_size = 1000
        # batch_size = 1
        # eps_init = 1.0
        # eps_end = 0.05
        # eps_num_steps = 10
        # gamma = 0.99
        # tau = 0.005
        # lr = 5e-5
        # max_grad_norm = 40
        # n_epochs = 1
        # max_steps = 1     # Steps run during eval

        shared_params = True

        if shared_params:
            self.qnet = self.build_q_agents_shared_params()
        else:
            self.qnet = self.build_q_agents()

        qnet_explore = TensorDictSequential(
            self.qnet,
            EGreedyModule(
                eps_init=eps_init,
                eps_end=eps_end,
                annealing_num_steps=eps_num_steps,
                action_key=("agents", "action"),
                spec=self.env.action_spec
            )
        )

        # qnet_explore = TensorDictSequential(
        #     self.qnet,
        #     CustomEGreedyModule(
        #         eps_init=eps_init,
        #         eps_end=eps_end,
        #         annealing_num_steps=eps_num_steps,
        #         action_key=("agents", "action"),
        #         spec=self.env.action_spec
        #     )
        # )

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

        # NOTE: DO NOT USE using cpu seemed to break training
        # collector = SyncDataCollector(
        #     self.env,
        #     qnet_explore,
        #     device="cpu",
        #     storing_device="cpu",
        #     policy_device="cpu",
        #     env_device="cpu",
        #     frames_per_batch=frames_per_collector_run,
        #     total_frames=total_frames
        # )

        collector = SyncDataCollector(
            self.env,
            qnet_explore,
            device=self.device,
            storing_device=self.device,
            policy_device=self.device,
            env_device=self.device,
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
        episode_rewards_moving_avg = []
        episode_percent_explored = []
        episode_percent_explored_moving_avg = []

        best_eval_reward = 0

        start_time_fname = time.strftime("%Y%m%d_%H%M%S")
        save_dir = "data/{}/".format(start_time_fname)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

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
            print("  new eps: {}".format(qnet_explore[1].eps))

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
                    policy=self.qnet,
                    auto_cast_to_device=True,
                    break_when_any_done=False
                )

                eval_reward = rollouts["next", "episode_reward"][-1].item()
                eval_percent_explored = rollouts["next", "percent_explored"][-1].item() * 100
                print("  eval reward: {}".format(eval_reward))

                episode_rewards.append(eval_reward)
                episode_percent_explored.append(eval_percent_explored)
                
                moving_avg_steps = 10
                reward_tot = 0
                percent_tot = 0
                moving_div = 0

                for iter_reward in range(1, min(len(episode_rewards), moving_avg_steps)+1):
                    reward_tot += episode_rewards[-iter_reward]
                    percent_tot += episode_percent_explored[-iter_reward]
                    moving_div += 1

                reward_avg = reward_tot / moving_div
                percent_avg = percent_tot / moving_div

                episode_rewards_moving_avg.append(reward_avg)
                episode_percent_explored_moving_avg.append(percent_avg)

                self.save_actions_to_file("{}/eval_{}_iter_{}.json".format(save_dir, start_time_fname, i), rollouts["agents", "action"])

                # TODO: change this to only save the state_dict?
                # Save qnet model
                if eval_reward > best_eval_reward:
                    torch.save(self.qnet, "{}/qnet_pickle_{}.pkl".format(save_dir, start_time_fname))
                    best_eval_reward = eval_reward

                evaluation_time = time.time() - evaluation_start

                print("  eval_time: {}".format(evaluation_time))

            sampling_start = time.time()

        total_time = time.time() - start_time
        print("total_time: {}".format(total_time))

        # Visualize last eval rollout
        if 1:
            vis = Visualizer()
            vis.init_game_vis()
            self.env_test.reset()
            vis.visualize_action_set(self.env_test, rollouts["agents", "action"])

        data = {
            "n_agents" :                            self.n_agents,
            "agent_abilities" :                     self.agent_abilities,
            "seed" :                                self.seed,
            "collector_runs" :                      collector_runs,
            "frames_per_collector_run" :            frames_per_collector_run,
            "total_frames" :                        total_frames,
            "memory_size" :                         memory_size,
            "batch_size" :                          batch_size,
            "eps_init" :                            eps_init,
            "eps_end" :                             eps_end,
            "eps_num_steps" :                       eps_num_steps,
            "gamma" :                               gamma,
            "tau" :                                 tau,
            "lr" :                                  lr,
            "max_grad_norm" :                       max_grad_norm,
            "n_epochs" :                            n_epochs,
            "max_steps" :                           max_steps,
            "shared_params" :                       shared_params,
            "total_time" :                          total_time,
            "episode_rewards" :                     episode_rewards,
            "episode_rewards_moving_avg" :          episode_rewards_moving_avg,
            "episode_percent_explored" :            episode_percent_explored,
            "episode_percent_explored_mvg_avg" :    episode_percent_explored_moving_avg
        }

        self.save_data_to_file("{}/data_{}.json".format(save_dir, start_time_fname), data)

        fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(10,8))
        # Plot reward per eval iteration
        ln1 = axs[0].plot(episode_rewards, label="Reward", color="blue")
        axs[0].set_title("Reward over iterations")
        axs[0].set_ylabel("Reward")
        axs[0].tick_params(axis="y")
        # Plot percent of map explored per eval iteration (on same plot as reward)
        axs_01 = axs[0].twinx()
        ln2 = axs_01.plot(episode_percent_explored, label="% explored", color="red")
        axs_01.set_ylabel("% explored")
        axs_01.tick_params(axis="y")
        axs_01.set_ylim([0, 100])
        # Plot average moving reward
        ln3 = axs[1].plot(episode_rewards_moving_avg, label="Avg Reward", color="blue")
        axs[1].set_title("Avg moving reward over {} iteration".format(moving_avg_steps))
        axs[1].set_ylabel("Reward")
        axs[1].tick_params(axis="y")
        # Plot average moving percent of map explored
        axs_02 = axs[1].twinx()
        ln4 = axs_02.plot(episode_percent_explored_moving_avg, label="Avg % explored", color="red")
        axs_02.set_ylabel("% explored")
        axs_02.tick_params(axis="y")
        axs_02.set_ylim([0, 100])
        
        lns1 = ln1+ln2
        labs1 = [l.get_label() for l in lns1]
        lns2 = ln3+ln4
        labs2 = [l.get_label() for l in lns1]
        axs[0].legend(lns1, labs1)
        axs[1].legend(lns2, labs2)

        for ax in axs.flat:
            ax.set(xlabel='Iterations', ylabel='Reward')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        fig.tight_layout()
        plt.savefig("{}/plot_{}".format(save_dir, start_time_fname))
        plt.show()

    # Save tensor of actions to json file
    def save_actions_to_file(self, fname, actions):
        # for step in actions:
        #     print("step: {}".format(step))
        list_actions = [step.tolist() for step in actions]
        dict_actions = {"seed" : self.seed, "n_agents" : self.n_agents, "agent_abilities" : self.agent_abilities, "actions" : list_actions}

        with open(fname, 'w') as fp:
            json.dump(dict_actions, fp)

    def save_data_to_file(self, fname, data):
        with open(fname, 'w') as fp:
            json.dump(data, fp)




if __name__ == "__main__":
    te = TeamExplore()

    # Single agent
    # te = TeamExplore(n_agents=1, agent_abilities=[[1, 3, 4]])

    te.train()
