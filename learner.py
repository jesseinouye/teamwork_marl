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


# I think this is the right order?
# Still not sure where reward comes in to play
#   - I think it gets piped into the loss module during training?

# cnn_module = MultiAgentConvNet > TensorDictModule
# mlp_module = MultiAgentMLP > TensorDictModule
# val_module = QValueModule
# qnet = SafeSequential(cnn_module, mlp_module, val_module)
# mixer = TensorDictModule(QMixer())
# loss_module = QMixerLoss(qnet, mixer)




# ==================================================================================
# Testing

class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.features = nn.Sequential(
            # Define according to the steps described above
            # Layer 1
            nn.Conv2d(3, 64, 3, stride=2, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Layer 2
            nn.Conv2d(64, 192, 3, stride=1, padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Layer 3
            nn.Conv2d(192, 384, 3, stride=1, padding=(1,1)),
            nn.ReLU(inplace=True),
            # # Layer 4
            # nn.Conv2d(384, 256, 3, stride=1, padding=(1,1)),
            # nn.ReLU(inplace=True),
            # # Layer 5
            # nn.Conv2d(256, 256, 3, stride=1, padding=(1,1)),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            # define according to the steps described above
            # Layer 1
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            # Layer 2
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Layer 3
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

def tmp_func():
    model = AlexNet(output_dim=10)

    print(summary(model,(3,32,32)))


def test_func():
    n_agents = 2
    env = EnvEngine(n_agents=n_agents, agent_abilities=[[1, 3], [1, 4]])

    print("obs spec: {}".format(env.observation_spec))


# ==================================================================================
# Actual code

def train():
    n_agents = 2

    hidden_dim = 4096   # TODO: change this to match flattened output from cnn
    action_space = 5    # TODO: change this so it pulls from env.action_spec.space.n ?

    seed = 0

    # Params from config?
    episodes = 50        # TODO: is this naming correct? should probably be something like "collections"
    batch_size = 512      # TODO: big powers of 2
    frames_per_episode = 4096
    total_frames = frames_per_episode * episodes
    memory_size = 100000         # TODO: increase this
    gamma = 0.95
    tau = 0.005
    lr = 5e-6
    # lr = 1e-3
    max_grad_norm = 40
    n_epochs = 10
    max_steps = 50     # Steps run during eval


    # Device
    device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    print("Using device: {}".format(device))

    torch.manual_seed(seed)

    # Set up environment
    env = EnvEngine(n_agents=n_agents, agent_abilities=[[1, 3], [1, 4]], map_size=32, device=device, seed=seed, max_steps=1024)
    env_test = EnvEngine(n_agents=n_agents, agent_abilities=[[1, 3], [1, 4]], map_size=32, device=device, seed=seed, max_steps=1024)

    # env = EnvEngine(n_agents=n_agents, agent_abilities=[[1, 3], [1, 4]], map_size=32, device=device, seed=seed, fname="simple_map.csv")
    # env_test = EnvEngine(n_agents=n_agents, agent_abilities=[[1, 3], [1, 4]], map_size=32, device=device, seed=seed, fname="simple_map.csv")

    cnn = MultiAgentConvNet(
        n_agents=n_agents,
        centralised=False,
        share_params=False,
        in_features=1,
        kernel_sizes=[5, 3, 3],
        num_cells=[32, 64, 64],
        strides=[2, 2, 1],
        paddings=[1, 1, 1],
        activation_class=torch.nn.ReLU,
        device=device
    )

    cnn_module = TensorDictModule(cnn, in_keys=[("agents", "observation")], out_keys=[("agents", "hidden")])

    mlp = MultiAgentMLP(
        n_agent_inputs=hidden_dim,
        n_agent_outputs=action_space,
        n_agents=n_agents,
        centralised=False,
        share_params=False,
        depth=2,
        num_cells=256,     # TODO: reduce this
        activation_class=nn.ReLU,
        device=device
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
        spec=env.action_spec,
        action_space="one-hot"
    )

    
    # qnet = SafeSequential(cnn_module, mlp_module, value_module)
    qnet = SafeSequential(cnn_module, mlp_module, softmax, value_module)

    print("Building: qnet_explore")
    qnet_explore = TensorDictSequential(
        qnet,
        EGreedyModule(
            eps_init=0.5,
            eps_end=0.1,
            annealing_num_steps=1000,
            action_key=("agents", "action"),
            spec=env.action_spec
        )
    )

    mixer = TensorDictModule(
        module=QMixer(
            state_shape=env.observation_spec["state"].shape,
            mixing_embed_dim=32,
            n_agents=n_agents,
            device=device
        ),
        in_keys=[("agents", "chosen_action_value"), "state"],
        out_keys=["chosen_action_value"]
    )

    print("Building: collector")
    collector = SyncDataCollector(
        env,
        qnet_explore,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_episode,
        total_frames=total_frames
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(memory_size, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=batch_size
    )

    loss_module = QMixerLoss(qnet, mixer, action_space="one-hot", delay_value=True)
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


    total_frames = 0
    sampling_start = time.time()
    start_time = time.time()

    print("Enumerating collector")
    for i, tensordict_data in enumerate(collector):
        print(f"ITERATION: {i}")

        sampling_time = time.time() - sampling_start
        print("  sampling_time: {}".format(sampling_time))

        # TODO: fix this? I think episode_reward should be total accumulated reward from entire episode
        # Set episode_reward to be the same as reward
        # tensordict_data.set(("next", "episode_reward"), tensordict_data.get(("next", "reward")))

        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        # print("current_frames: {}".format(current_frames))


        training_tds = []
        training_start = time.time()
        for j in range(n_epochs):
            # print("EPOCH: {}".format(j))
            for k in range(frames_per_episode // batch_size):
                # print("BATCH: {}".format(k))
                subdata = replay_buffer.sample()

                # print("subdata: {}".format(subdata))

                loss_vals = loss_module(subdata)
                # training_tds.append(loss_vals.detach())

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

                # Printing module weights to test that they're changing
                # print("module: {}".format(cnn_module._modules["module"]._modules["agent_networks"][0][0].weight))

        qnet_explore[1].step(frames=current_frames)
        collector.update_policy_weights_()

        training_time = time.time() - training_start

        print("  training_time: {}".format(training_time))

        iteration_time = sampling_time + training_time
        
        training_tds = torch.stack(training_tds)

        # print("module: {}".format(cnn_module._modules["module"]._modules["agent_networks"][0][0].weight))
        # print("module: {}".format(mlp_module._modules["module"]._modules["agent_networks"]))


        # print("Evaluating")
        evaluation_start = time.time()
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        # with torch.no_grad():
            env_test.frames = []
            rollouts = env_test.rollout(
                max_steps=max_steps,
                policy=qnet_explore,
                auto_cast_to_device=True,
                break_when_any_done=False
            )

            evaluation_time = time.time() - evaluation_start

            print("  eval_time: {}".format(evaluation_time))



        # print("rollout:\n{}".format(rollouts))

        # TODO: is it right to reset here? Otherwise the collector doesn't reset the env before it collects more samples?
        env.reset()
        sampling_start = time.time()

    print("total_time: {}".format(time.time() - start_time))

    # Visualizer
    if 1:
        vis = Visualizer()
        vis.init_game_vis()
        env_test.reset()
        vis.visualize_action_set(env_test, rollouts["agents", "action"])


    # print("loss:\n{}".format(training_tds["loss"]))
    # print("grad_norm:\n{}".format(training_tds["grad_norm"]))


    # plt.plot(training_tds["loss"].cpu().detach().numpy())
    # plt.savefig('loss.png')
    # plt.show()



    # TODO: left off here:
    #           - Build out qnet (cnn_module > mlp_module > value_module in safe sequetial)
    #           - Build out q mixer ?


    # -----------------------------------------------------
    # Testing
    if 0:
        batch = 1
        channels, x, y = 1, 32, 32

        # Test cnn
        obs = torch.randn(batch, n_agents, channels, x, y)
        print("cnn:\n{}".format(cnn))
        result = cnn(obs)
        print("result: {}".format(result.shape))

        # Test mlp
        obs = torch.randn(batch, n_agents, hidden_dim)
        print("mlp:\n{}".format(mlp))
        result = mlp(obs)
        print("result: {}".format(result.shape))






if __name__ == "__main__":
    
    train()

    # tmp_func()

    # test_func()