import sys
import json
import pygame
import random
import torch

from torchrl.envs.utils import check_env_specs

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential

from environment_engine import EnvEngine, Action, Agent
from tile import CellType



WIDTH = 800
HEIGHT = 800
FULL_WIDTH = WIDTH * 2
BLACK = (0, 0, 0)
GRAY = (125, 125, 125)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PINK = (255, 0, 255)
PURPLE = (150, 0, 200)
ORANGE = (245, 146, 24)
GRAY_ORANGE = (61, 55, 41)

class Visualizer():
    def __init__(self) -> None:
        self.device = "cpu"

        # self.map = self.env.generate_map()
        # self.env.load_agent(abilities=[1, 3])
        # self.env.load_agent(abilities=[1, 4])
        # self.env.place_agents_at_start()
        # self.env.load_agent(abilities=[1,2])
        # self.env.place_agents()

        self.color_map = {
            CellType.WALL: BLACK,
            CellType.GRASS: GREEN,
            CellType.WATER: BLUE,
            CellType.FLOOR: WHITE,
            CellType.AGENT_1: RED,
            CellType.AGENT_2: PINK,
            CellType.AGENT_3: PURPLE,
            CellType.AGENT_4: ORANGE,
            CellType.NULL: GRAY_ORANGE
            # Add other CellType mappings here
        }

        pygame.init()

    def init_env(self, seed=None, fname=None, n_agents=2, agent_abilities=[[1, 3], [1, 4]]):
        # self.env = EnvEngine(n_agents=2, agent_abilities=[[1, 3], [1, 4]], seed=seed, fname=fname)
        self.env = EnvEngine(n_agents=n_agents, agent_abilities=agent_abilities, seed=seed, fname=fname)
        
        # Params
        self.rows = self.env.rows
        self.cols = self.env.cols

        self.cell_size = WIDTH // self.cols

    def init_game_vis(self):
        self.screen = pygame.display.set_mode((FULL_WIDTH, HEIGHT))
        pygame.display.set_caption("SLAM Visualizer")
        self.screen.fill(WHITE)


    def draw_map(self, screen, map, agents:list[Agent]=[]):
        # Draw ground truth map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_type = map[row][col].get_type()
                cell_color = self.color_map.get(cell_type, WHITE)  # Default to WHITE if not found
                pygame.draw.rect(screen, cell_color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

        # Draw agent locations
        # if agents:
        #     for agent in agents:
        #         if agent.position is not None:
        #             pygame.draw.rect(screen, RED, (agent.position[1] * self.cell_size, agent.position[0] * self.cell_size, self.cell_size, self.cell_size))

    def draw_observation_map(self, screen, obs_map, agents:list[Agent]=[]):
        # Draw observation map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_obs_type = obs_map[row][col].get_type()
                cell_obs_color = self.color_map.get(cell_obs_type, GRAY)
                pygame.draw.rect(screen, cell_obs_color, (WIDTH + (col * self.cell_size), row * self.cell_size, self.cell_size, self.cell_size))

        # Draw agent locations
        # if agents:
        #     for agent in agents:
        #         if agent.position is not None:
        #             pygame.draw.rect(screen, RED, (WIDTH + (agent.position[1] * self.cell_size), agent.position[0] * self.cell_size, self.cell_size, self.cell_size))


    def draw_map_no_agents(self, screen, map):
        # Draw ground truth map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_type = map[row][col].get_type()
                cell_color = self.color_map.get(cell_type, WHITE)  # Default to WHITE if not found
                pygame.draw.rect(screen, cell_color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

    def draw_observation_map_no_agents(self, screen, obs_map):
        # Draw observation map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_obs_type = obs_map[row][col].get_type()
                cell_obs_color = self.color_map.get(cell_obs_type, GRAY)
                pygame.draw.rect(screen, cell_obs_color, (WIDTH + (col * self.cell_size), row * self.cell_size, self.cell_size, self.cell_size))

                    
    def draw_map_vis(self, screen, map):
        # Draw ground truth map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_type = map[row][col]
                cell_color = self.color_map.get(cell_type, WHITE)  # Default to WHITE if not found
                pygame.draw.rect(screen, cell_color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

    def draw_observation_map_vis(self, screen, obs_map):
        # Draw observation map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_obs_type = obs_map[row][col]
                cell_obs_color = self.color_map.get(cell_obs_type, GRAY)
                pygame.draw.rect(screen, cell_obs_color, (WIDTH + (col * self.cell_size), row * self.cell_size, self.cell_size, self.cell_size))



    def main(self):
        screen = pygame.display.set_mode((FULL_WIDTH, HEIGHT))
        pygame.display.set_caption("SLAM Visualizer")
        screen.fill(WHITE)

        self.init_env()

        running = True

        # Initialize the agent's direction, assuming it starts facing EAST
        self.agent_direction = Action.EAST
        agents = self.env.get_agents()
        agent = agents[0]  # Assuming there's at least one agent

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return

            # Record the agent's current position
            last_position = agent.position

            # Autonomous exploration
            # TODO: change this to use calc_agent_observations, but use direction somehow?
            obs = self.env.agent_peek(agent, self.agent_direction)  # Get the observation in front of the agent
            
            # Move forward if possible
            if self.can_move_forward(obs):
                self.env.move_agent(agent, self.agent_direction)
            else:
                self.agent_direction = self.choose_new_direction(self.agent_direction)

            # Check if the agent has moved
            if agent.position == last_position:
                # The agent hasn't moved; it might be facing an obstacle.
                self.agent_direction = self.choose_new_direction(self.agent_direction)
                self.env.move_agent(agent, self.agent_direction)

            map = self.env.get_map()
            obs_map = self.env.get_obs_map()
            self.draw_map(screen, map, agents)
            self.draw_observation_map(screen, obs_map, agents)
            pygame.display.update()

            # Adding a small delay can make the agent's movement easier to observe
            pygame.time.delay(500)
    
    def can_move_forward(self, obs):
        # Check if the agent can move forward based on the observation
        print("obs: {}".format(obs))
        if len(obs) == 0:
            return False
        return (obs[0]["type"] != CellType.WALL and obs[0]["type"] != CellType.OOB)
    

    def choose_new_direction(self, current_direction):
        # Randomly choose to turn left or right
        directions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
        directions.remove(current_direction)
        return random.choice(directions)
    



    def test_main(self):
        screen = pygame.display.set_mode((FULL_WIDTH, HEIGHT))
        pygame.display.set_caption("SLAM Visualizer")
        screen.fill(WHITE)

        # self.init_env(seed=0)
        # self.init_env(seed=0, fname="simple_map.csv")
        # self.init_env(seed=0, fname="test_map.csv")
        self.init_env(seed=0, n_agents=1, agent_abilities=[[1,3,4]])

        running = True

        # Reset env to start
        self.env.reset()

        # Test actions w/ 2 agents - agent 1 down, agent 2 left
        actions = TensorDict(
            {"agents": TensorDict(
                {"action": torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])},
                batch_size=(),
                device=self.device)
            },
            batch_size=(),
            device=self.device
        )

        # Test actions w/ 4 agents
        # actions = TensorDict(
        #     {"agents": TensorDict(
        #         {"action": torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]])},
        #         batch_size=(),
        #         device=self.device)
        #     },
        #     batch_size=(),
        #     device=self.device
        # )

        # Test actions w/ 1 agent
        actions = TensorDict(
            {"agents": TensorDict(
                {"action": torch.tensor([[0, 0, 1, 0, 0]])},
                batch_size=(),
                device=self.device)
            },
            batch_size=(),
            device=self.device
        )

        i = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return

            if i < 50:
            # if True:
                if i > 2:
                    actions = TensorDict(
                        {"agents": TensorDict(
                            {"action": torch.tensor([[0, 0, 0, 1, 0]])},
                            batch_size=(),
                            device=self.device)
                        },
                        batch_size=(),
                        device=self.device
                    )
                # Perform step in env
                self.env.step(actions)

                # print("actions:\n{}".format(actions))
                print("reward: {}".format(actions["next", "reward"][0]))

                # Get observation (map) from output of step
                obs_map = actions["next", "agents", "observation"]
                obs_map = obs_map[0,0].numpy()

                # obs_map = actions["next", "local_obs"]
                # obs_map = obs_map.numpy()

                # Get ground truth state (map) from output of step
                map = actions["next", "state"]
                map = map.numpy()

                i += 1

            # Draw ground truth and observation maps
            self.draw_map_vis(screen, map)
            self.draw_observation_map_vis(screen, obs_map)
            
            # Update pygame display
            # Adding a small delay can make the agent's movement easier to observe
            pygame.display.update()
            pygame.time.delay(100)



        # self.env.reset(None)

    def visualize_action_set(self, env:EnvEngine, actions:torch.Tensor):
        self.init_game_vis()

        self.rows = env.rows
        self.cols = env.cols

        self.cell_size = WIDTH // self.cols

        env.reset()

        for action in actions:
            # action = torch.unsqueeze(action, 0)

            print("action: {}".format(action))

            action = TensorDict(
                {"agents": TensorDict(
                    {"action": action},
                    batch_size=(),
                    device=self.device)
                },
                batch_size=(),
                device=self.device
            )

            env.step(action)

            # print("actions:\n{}".format(actions))
            # print("reward: {}".format(action["next", "reward"][0]))

            # Get observation (map) from output of step
            obs_map = action["next", "agents", "observation"]
            obs_map = obs_map[0,0].numpy()

            # Get ground truth state (map) from output of step
            map = action["next", "state"]
            map = map.numpy()

            # Draw ground truth and observation maps
            self.draw_map_vis(self.screen, map)
            self.draw_observation_map_vis(self.screen, obs_map)
            
            # Update pygame display
            # Adding a small delay can make the agent's movement easier to observe
            pygame.display.update()
            pygame.time.delay(200)


    def vis_from_file_playback(self, fname):
        self.init_game_vis()

        with open(fname) as f:
            data = json.load(f)

        print("data: {}".format(data))

        n_agents = 2
        agent_abilities = [[1, 3], [1, 4]]

        if "n_agents" in data:
            n_agents = data["n_agents"]
        if "agent_abilities" in data:
            agent_abilities = data["agent_abilities"]


        self.init_env(seed=data["seed"], n_agents=n_agents, agent_abilities=agent_abilities)

        running = True

        # Reset env to start
        self.env.reset()

        actions = data["actions"]

        for action in actions:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                
            action = torch.tensor(action)

            print("actions: {}".format(action))

            action = TensorDict(
                {"agents": TensorDict(
                    {"action": action},
                    batch_size=(),
                    device=self.device)
                },
                batch_size=(),
                device=self.device
            )
                
            self.env.step(action)

            # print("actions:\n{}".format(actions))
            print("reward: {}".format(action["next", "reward"][0]))

            # Get observation (map) from output of step
            obs_map = action["next", "agents", "observation"]
            obs_map = obs_map[0,0].numpy()

            # Get ground truth state (map) from output of step
            map = action["next", "state"]
            map = map.numpy()

            # Draw ground truth and observation maps
            self.draw_map_vis(self.screen, map)
            self.draw_observation_map_vis(self.screen, obs_map)
            
            # Update pygame display
            # Adding a small delay can make the agent's movement easier to observe
            pygame.display.update()
            pygame.time.delay(100)

        print("Episode reward: {}".format(action["next", "episode_reward"].item()))

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
            
            pygame.display.update()
            pygame.time.delay(250)


    def map_test(self):
        screen = pygame.display.set_mode((FULL_WIDTH, HEIGHT))
        pygame.display.set_caption("SLAM Visualizer")
        screen.fill(WHITE)

        self.init_env(seed=0)

        fname = "test_map.csv"

        self.env.save_map(fname)




if __name__ == "__main__":
    vis = Visualizer()
    # vis.main()

    # vis.test_main()

    # vis.map_test()

    # fname = "eval_20240429-130711_iter_0.json"

    # fname = "eval_20240501-122636_iter_46.json"
    # fname = "eval_20240501-122624_iter_45.json"

    # fname = "eval_20240501-125055_iter_33.json"

    # fname = "eval_20240501-141132_iter_41.json"
    # fname = "eval_20240501-141216_iter_43.json"

    # fname = "eval_20240501-143507_iter_19.json"

    # fname = "eval_20240502-131545_iter_30.json"
    # fname = "eval_20240502-144646_iter_116.json"

    # fname = "eval_20240502_155058_iter_86.json"

    # vis.vis_from_file_playback(fname)

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        print("Playing file: {}".format(fname))
        vis.vis_from_file_playback(fname)
    else:
        print("No filename given, playing test action")
        vis.test_main()


            
