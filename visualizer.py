import pygame
import random
import torch

from torchrl.envs.utils import check_env_specs

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential

from environment_engine import EnvEngine, Action, Agent, CellType



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

class Visualizer():
    def __init__(self) -> None:
        self.device = "cpu"

        self.env = EnvEngine(n_agents=2, agent_abilities=[[1, 3], [1, 4]])
        # self.env = EnvEngine(n_agents=4, agent_abilities=[[1, 3], [1, 4], [1], [1]])
        

        # Params
        self.rows = self.env.rows
        self.cols = self.env.cols

        self.cell_size = WIDTH // self.cols

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
            CellType.AGENT_4: ORANGE
            # Add other CellType mappings here
        }

        pygame.init()



    def draw_map(self, screen, map, agents:list[Agent]=[]):
        # Draw ground truth map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_type = map[row][col]
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
                cell_obs_type = obs_map[row][col]
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
                cell_type = map[row][col]
                cell_color = self.color_map.get(cell_type, WHITE)  # Default to WHITE if not found
                pygame.draw.rect(screen, cell_color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

    def draw_observation_map_no_agents(self, screen, obs_map):
        # Draw observation map
        for row in range(self.rows):
            for col in range(self.cols):
                cell_obs_type = obs_map[row][col]
                cell_obs_color = self.color_map.get(cell_obs_type, GRAY)
                pygame.draw.rect(screen, cell_obs_color, (WIDTH + (col * self.cell_size), row * self.cell_size, self.cell_size, self.cell_size))



    # your original main: 
                    
    # def main(self):
    #     screen = pygame.display.set_mode((WIDTH, HEIGHT))
    #     pygame.display.set_caption("SLAM Visualizer")
    #     screen.fill(WHITE)

    #     running = True
    #     i = 0
    #     while running:
    #         # print("running iter: {}".format(i))
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False
    #                 pygame.quit()
    #                 return
    #             if event.type == pygame.KEYDOWN:
    #                 print("got keydown")
    #                 if event.key == pygame.K_LEFT:
    #                     self.env.move_agent(agents[0], Action.WEST)
    #                 if event.key == pygame.K_RIGHT:
    #                     self.env.move_agent(agents[0], Action.EAST)
    #                 if event.key == pygame.K_UP:
    #                     self.env.move_agent(agents[0], Action.NORTH)
    #                 if event.key == pygame.K_DOWN:
    #                     self.env.move_agent(agents[0], Action.SOUTH)
                    
    #                 if event.key == pygame.K_SPACE:
    #                     print("got space")
    #                     self.env.place_agents()
    #                     obs = self.env.calc_agent_observation(agents[0])
    #                     print("obs: {}".format(obs))

    #         map = self.env.get_map()
    #         agents = self.env.get_agents()

    #         self.draw_map(screen, map, agents)
    #         pygame.display.update()
    #         # pygame.display.flip()
    #         # i += 1
                    

    def main(self):
        screen = pygame.display.set_mode((FULL_WIDTH, HEIGHT))
        pygame.display.set_caption("SLAM Visualizer")
        screen.fill(WHITE)

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
        return (obs[0]['type'] != CellType.WALL and obs[0]['type'] != CellType.OOB)
    

    def choose_new_direction(self, current_direction):
        # Randomly choose to turn left or right
        directions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
        directions.remove(current_direction)
        return random.choice(directions)
    



    def test_main(self):
        screen = pygame.display.set_mode((FULL_WIDTH, HEIGHT))
        pygame.display.set_caption("SLAM Visualizer")
        screen.fill(WHITE)

        running = True

        # Reset env to start
        self.env.reset()

        # Test actions w/ 2 agents - agent 1 down, agent 2 left
        actions = TensorDict(
            {"agents": TensorDict(
                {"action": torch.tensor([[0, 0, 0, 1, 0], [0, 0, 1, 0, 0]])},
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

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return

            # Perform step in env
            self.env.step(actions)

            print("reward: {}".format(actions["next", "reward"][0]))

            # Get observation (map) from output of step
            obs_map = actions["next", "agents", "observation"]
            obs_map = obs_map[0].numpy()

            # Get ground truth state (map) from output of step
            map = actions["next", "state"]
            map = map.numpy()

            # Draw ground truth and observation maps
            self.draw_map_no_agents(screen, map)
            self.draw_observation_map_no_agents(screen, obs_map)
            
            # Update pygame display
            # Adding a small delay can make the agent's movement easier to observe
            pygame.display.update()
            pygame.time.delay(500)



        # self.env.reset(None)



if __name__ == "__main__":
    vis = Visualizer()
    # vis.main()

    vis.test_main()
            
