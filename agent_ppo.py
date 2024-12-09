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
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
# from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, QValueModule, SafeSequential, DistributionalDQNnet, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import MultiAgentMLP, QMixer, MultiAgentConvNet
from torchrl.objectives import SoftUpdate, ValueEstimators, ClipPPOLoss
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

        self.build_envs()

    def build_envs(self):
        self.env = EnvEngine(n_agents=self.n_agents, agent_abilities=self.agent_abilities, map_size=32, device=self.device, seed=self.seed, max_steps=512)
        self.env_test = EnvEngine(n_agents=self.n_agents, agent_abilities=self.agent_abilities, map_size=32, device=self.device, seed=self.seed, max_steps=512)

    def build_agents(self):
        hidden_dim = 4096
        action_space = 5

        cnn = MultiAgentConvNet(
            n_agents=self.n_agents,
            centralized=False,
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

        mlp = nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=hidden_dim,
                n_agent_outputs=2*action_space,
                n_agents=self.n_agents,
                centralized=False,
                share_params=False,
                depth=2,
                num_cells=256,
                activation_class=nn.ReLU,
                device=self.device
            ),
            NormalParamExtractor()
        )

        mlp_module = TensorDictModule(mlp, in_keys=[("agents", "hidden")], out_keys=[("agents", "loc"), ("agents", "scale")])

        actor_module = SafeSequential(cnn_module, mlp_module)

        # NOTE: might need to figure out what "low" and "high" are for our env
        #       seems like these are min and max values of a distribution based on our env...?
        #       see: https://pytorch.org/rl/stable/tutorials/multiagent_ppo.html
        policy = ProbabilisticActor(
            module=actor_module,
            spec=self.env.action_spec,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[self.env.action_key],
            distribution_class=TanhNormal,
            # distribution_kwargs={
            #     "low": self.env.action_spec[("agents", "action")].space.low,
            #     "high": self.env.action_spec[("agents", "action")].space.high
            # },
            distribution_kwargs={
                "low": 0,
                "high": 1
            },
            return_log_prob=True
        )

        return policy

    def build_critic(self):
        hidden_dim = 4096
        action_space = 5

        # cnn = MultiAgentConvNet(
        #     n_agents=self.n_agents,
        #     centralized=False,
        #     share_params=False,
        #     in_features=1,
        #     kernel_sizes=[5, 3, 3],
        #     num_cells=[32, 64, 64],
        #     strides=[2, 2, 1],
        #     paddings=[1, 1, 1],
        #     activation_class=torch.nn.ReLU,
        #     device=self.device
        # )
        
        # cnn_module = TensorDictModule(cnn, in_keys=[("agents", "observation")], out_keys=[("agents", "hidden")])

        # mlp = MultiAgentMLP(
        #         n_agent_inputs=hidden_dim,
        #         n_agent_outputs=1,
        #         n_agents=self.n_agents,
        #         centralized=True,
        #         share_params=False,
        #         depth=2,
        #         num_cells=256,
        #         activation_class=nn.ReLU,
        #         device=self.device
        #     )

        # mlp_module = TensorDictModule(mlp, in_keys=[("agents", "hidden")])

        # critic_module = SafeSequential(cnn_module, mlp_module)

        critic_module = nn.Sequential(
            MultiAgentConvNet(
                n_agents=self.n_agents,
                centralized=True,
                share_params=False,
                in_features=1,
                kernel_sizes=[5, 3, 3],
                num_cells=[32, 64, 64],
                strides=[2, 2, 1],
                paddings=[1, 1, 1],
                activation_class=torch.nn.ReLU,
                device=self.device
            ),
            MultiAgentMLP(
                n_agent_inputs=hidden_dim,
                n_agent_outputs=1,
                n_agents=self.n_agents,
                centralized=True,
                share_params=False,
                depth=2,
                num_cells=256,
                activation_class=nn.ReLU,
                device=self.device
            )
        )

        value_module = ValueOperator(
            module=critic_module,
            in_keys=[("agents", "state")]
            # in_keys=[("agents", "observation")]
        )
        
        return value_module

    def train(self):

        # Tmp training params
        collector_runs = 100
        # frames_per_collector_run = 8192
        frames_per_collector_run = 4096
        total_frames = frames_per_collector_run * collector_runs   
        memory_size = 50000
        batch_size = 512
        eps_init = 1.0
        eps_end = 0.05
        eps_num_steps = 100000
        gamma = 0.99
        lmbda = 0.9
        tau = 0.005
        lr = 5e-5
        max_grad_norm = 30
        n_epochs = 5
        max_steps = 300     # Steps run during eval
    
        # PPO stuff
        clip_epsilon = 0.2
        entropy_eps = 1e-3
        # entropy_eps = 1e-4

        shared_params = False

        self.policy = self.build_agents()
        self.value = self.build_critic()

        collector = SyncDataCollector(
            self.env,
            self.policy,
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

        loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.value,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_eps,
            normalize_advantage=False
        )
        loss_module.set_keys(
            # reward=self.env.reward_key,
            # action=self.env.action_key,
            reward=("agents", "reward"),
            action=("agents", "action"),
            done=("agents", "done"),
            terminated=("agents", "terminated")
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
        )

        optim = torch.optim.Adam(loss_module.parameters(), lr=lr)

        episode_rewards = []
        episode_rewards_moving_avg = []
        episode_percent_explored = []
        episode_percent_explored_moving_avg = []

        best_eval_reward = 0

        start_time_fname = time.strftime("%Y%m%d_%H%M%S")
        save_dir = "data/{}/".format(start_time_fname)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        eval_dir = "{}/eval/".format(save_dir)
        Path(eval_dir).mkdir(parents=True, exist_ok=True)


        total_frames = 0
        start_time = time.time()
        sampling_time = time.time()

        print("Enumerating collector")
        for i, tensordict_data in enumerate(collector):
            print(f"ITERATION: {i}")

            sampling_time = time.time() - start_time
            print("  sampling_time: {}".format(sampling_time))

            with torch.no_grad():
                loss_module.value_estimator(
                    tensordict_data,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params
                )

            current_frames = tensordict_data.numel()
            total_frames += current_frames
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view)

            training_tds = []
            training_start = time.time()
            for _ in range(n_epochs):
                for _ in range(frames_per_collector_run // batch_size):
                    subdata = replay_buffer.sample()
                    loss_vals = loss_module(subdata)
                    training_tds.append(loss_vals.detach())

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()

                    total_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), max_grad_norm
                    )
                    training_tds[-1].set("grad_norm", total_norm.mean())

                    optim.step()
                    optim.zero_grad()

            collector.update_policy_weights_()
            
            training_time = time.time() - training_start
            print("  training_time: {}".format(training_time))

            iteration_time = sampling_time + training_time
            print("  iteration_time: {}".format(iteration_time))

            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                self.env_test.frames = []
                rollouts = self.env_test.rollout(
                    max_steps=max_steps,
                    policy=self.policy,
                    auto_cast_to_device=True,
                    break_when_any_done=False
                )

                eval_reward = rollouts["next", "episode_reward"][-1].item()
                eval_percent_explored = rollouts["next", "percent_explored"][-1].item() * 100
                print("  eval reward: {}  -- perc explored: {}".format(eval_reward, eval_percent_explored))

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

                self.save_actions_to_file("{}/eval_{}_iter_{}.json".format(eval_dir, start_time_fname, i), rollouts["agents", "action"])

                if eval_reward > best_eval_reward:
                    torch.save(self.policy, "{}/policy_pickle_{}.pkl".format(save_dir, start_time_fname))
                    best_eval_reward = eval_reward

                evaluation_time = time.time() - evaluation_start
                print("  eval_time: {}".format(evaluation_time))

            sampling_time = time.time()

        total_time = time.time() - start_time
        print("total_time: {}".format(total_time))

        collector.shutdown()
        if not self.env.is_closed:
            self.env.close()
        if not self.env_test.is_closed:
            self.env_test.close()

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
    appo = AgentPPO()
    appo.train()