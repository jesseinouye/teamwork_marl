import sys
import random
import copy
import torch
import numpy as np
from typing import List, Optional
from enum import Enum, IntEnum
from collections import Counter

from tensordict import TensorDict, TensorDictBase

from torchrl.envs import EnvBase
from torchrl.data import DiscreteTensorSpec, BoundedTensorSpec, OneHotDiscreteTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec

from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.libs.smacv2 import SMACv2Env

from tile import CellType, Tile 


# Agent observations
# class AgentObs(CellType):
#     UNKNOWN = 6

np.set_printoptions(threshold=sys.maxsize)


# Actions agents can take
class Action(Enum):
    IDLE = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


# Agent class
class Agent():
    def __init__(self, id, abilities=[1], position=None):
        self.id = id
        self.abilities = abilities
        self.position = position
        self.rangeOfSight = 3
    
    def getPosition(self):
        return self.position


# Environment engine
class EnvEngine(EnvBase):

    def __init__(self, n_agents=2, device="cpu", map_size=32, agent_abilities=[[1], [1]], seed=None, fname=None) -> None:
        if seed is not None:
            self._set_seed(seed)

        self.map_data = []

        # initialize discovered tiles
        self.discovered_tiles_num = 0

        super().__init__(device=device, batch_size=[])
        
        # Parameters
        self.rows = map_size
        self.cols = map_size    # TODO: change this to allow rectangles?

        self.agent_obs_dist = 3

        if fname is None:
            # If no filename give, generate map
            self.map = self.generate_map()
        else:
            # If filename given, load map from file
            self.map = self.load_map(fname)

        # Create state map (contains agent locations)
        self.state_map = copy.deepcopy(self.map)

        # Create observation map
        self.obs_map = [[Tile(CellType.UNKNOWN) for _ in range(self.cols)] for _ in range(self.rows)]
        self.obs_map = np.array(self.obs_map)

        # List of agents
        self.agents  : List[Agent] = []
        
        self.n_agents = n_agents
        self.n_actions = len(Action)

        if len(agent_abilities) != n_agents:
            raise ValueError("ERROR: length of agent ability list (agent_abilities) must match number of agents (n_agents)")

        for i in range(n_agents):
            self.load_agent(abilities=agent_abilities[i])

        # TODO: figure out how to define this - determined by whether we use a CNN or flattened tensor?
        self.obs_size = self.rows * self.cols

        self.cur_step_reward = 0

        self.cur_step = 0

        # Make spec for action and observation
        self._make_spec()


    def _make_spec(self):
        self.action_spec = self._make_action_spec()
        self.observation_spec = self._make_observation_spec()

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=torch.Size((1,)),
            device=self.device
        )

        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=torch.Size((1,)),
            dtype=torch.int8,
            device=self.device
        )

    def _make_action_spec(self) -> CompositeSpec:
        action_spec = OneHotDiscreteTensorSpec(
            self.n_actions,
            shape=torch.Size((self.n_agents, self.n_actions)),
            device=self.device,
            dtype=torch.long,
        )

        full_action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": action_spec}, shape=torch.Size((self.n_agents,))
                )
            }
        )
        return full_action_spec
    
    def _make_observation_spec(self) -> CompositeSpec:
        obs_spec = DiscreteTensorSpec(
            n=len(CellType),
            # shape=torch.Size((self.n_agents, self.obs_size)),
            shape=torch.Size((self.n_agents, self.rows, self.cols)),
            dtype=torch.float32,
            device=self.device
        )
        info_spec = CompositeSpec(
            {
                "episode_limit": DiscreteTensorSpec(
                    2, dtype=torch.int8, device=self.device
                ),
                "map_explored": DiscreteTensorSpec(
                    2, dtype=torch.int8, device=self.device
                )
            }
        )
        mask_spec = DiscreteTensorSpec(
            2,
            torch.Size((self.n_agents, self.n_actions)),
            device=self.device,
            dtype=torch.int8
        )
        spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"observation": obs_spec, "action_mask": mask_spec},
                    shape=torch.Size((self.n_agents,))
                ),
                "state": DiscreteTensorSpec(
                    n=len(CellType),
                    # shape=torch.Size((self.n_agents, self.obs_size)),
                    shape=torch.Size((self.rows, self.cols)),
                    dtype=torch.float32,
                    device=self.device
                ),
                "info": info_spec
            }
        )
        return spec

    def _set_seed(self, seed: Optional[int]):
        print("Using seed: {}".format(seed))
        rng = torch.manual_seed(seed)
        self.rng = rng
        random.seed(seed)
        # TODO: set seed for map generation too?


    def _step(self, tensordict: TensorDictBase):
        # At each step:
        #   For each agent
        #   - Take agent action (figure out how all agent actions are passed in tensordict?)
        #   - Move agent based on action
        #   - Determine observation

        #   After all agent actions done
        #   - Calculate reward from observation and prev observed map?
        #   - Update observed map
        
        actions = tensordict["agents", "action"]

        # Get decoded movements from one-hot tensor
        acts = torch.argmax(actions, dim=1)

        all_agent_obs = []
        self.cur_step_reward = 0

        # Move each agent in order
        for i, action in enumerate(acts):
            # print("i: {} / action: {}".format(i, action))
            self.move_agent(self.agents[i], Action(action.item()))
            agent_obs = self.map_to_numeric(self.calc_agent_observation(self.agents[i]))
            all_agent_obs.append(agent_obs)

        all_agent_obs = np.array(all_agent_obs)
        obs = torch.tensor(all_agent_obs, dtype=torch.float32)
        
        # Adding 'channels' dimension
        obs = torch.unsqueeze(obs, 1)

        state = torch.tensor(self.map_to_numeric(self.state_map), dtype=torch.float32)

        reward = torch.tensor([self.cur_step_reward], device=self.device, dtype=torch.float32)

        # print("step -- actions: {} -- reward: {}".format(acts, reward))

        # TODO: Calculate if done
        done = torch.tensor([0])


        # TODO: fix these to output actual data
        #       should match format of observation spec ?

        # obs = torch.zeros(self.n_agents, self.obs_size)

        # TODO: remove action_mask from the spec (it's not needed - used to dynamically mask valid/invalid actions)
        mask = torch.tensor([[1, 1, 1, 1, 1]] * self.n_agents)

        # state = torch.zeros(self.n_agents, self.obs_size)

        agents_td = TensorDict(
            {"observation": obs, "action_mask": mask}, batch_size=(self.n_agents,)
        )

        info = TensorDict(
            source={
                "episode_limit": torch.tensor([100]),
                "map_explored": torch.tensor([0])
            },
            batch_size=(),
            device=self.device
        )

        out = TensorDict(
            source={
                "agents": agents_td,
                "state": state,
                "info": info,
                "reward": reward,
                "done": done,
                "terminated": done.clone()
            },
            batch_size=(),
            device=self.device
        )
        
        self.cur_step += 1
        # print("executed step: {} - actions were: {}".format(self.cur_step, acts))

        return out

    def _reset(self, tensordict:TensorDictBase):
        # Clear observed map
        # Move all agents to start (upper left corner)
        # print("resetting!!")

        self.cur_step = 0

        # Clear observed and state maps
        self.clear_obs_map()
        self.clear_state_map()
        
        # Move all agents to start
        self.place_agents_at_start()
        self.numeric_state_map = self.map_to_numeric(self.state_map)

        all_agent_obs = []

        for agent in self.agents:
            agent_obs = self.calc_agent_observation(agent)  # this should return a map
            numeric_agent_obs = self.map_to_numeric(agent_obs)
            all_agent_obs.append(numeric_agent_obs)  # all_agent_obs should be a list of 2D arrays

        # Convert all_agent_obs to a single numpy array if necessary, depends on your use-case
        # Example: If agents' observations can be stacked or concatenated
        # For demonstration, let's assume we stack them along a new axis
        obs_array = np.stack(all_agent_obs, axis=0)
        obs = torch.tensor(obs_array, dtype=torch.float32)
        state = torch.tensor(self.numeric_state_map, dtype=torch.float32)

        # Adding 'channels' dimension
        obs = torch.unsqueeze(obs, 1)

        # obs = torch.tensor(all_agent_obs)
        # state = torch.tensor(self.state_map)


        # TODO: fix these to output actual data
        #       should match format of observation spec ?

        # obs = torch.zeros(self.n_agents, self.obs_size)
        # tmp_mask = [[1, 0, 0, 0, 0]] * self.n_agents
        mask = torch.tensor([[1, 1, 1, 1, 1]] * self.n_agents)
        # mask = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])

        # state = torch.zeros(self.n_agents, self.obs_size)

        agents_td = TensorDict(
            {"observation": obs, "action_mask": mask}, batch_size=(self.n_agents,)
        )

        info = TensorDict(
            source={
                "episode_limit": torch.tensor([100]),
                "map_explored": torch.tensor([0])
            },
            batch_size=(),
            device=self.device
        )

        reward = torch.tensor([0])
        done = torch.tensor([0])

        out = TensorDict(
            source={
                "agents": agents_td,
                "state": state,
                "info": info,
                # "reward": reward,
                # "done": done,
                # "terminated": done.clone()
            },
            batch_size=(),
            device=self.device
        )

        # print("OBS: {}".format(obs_array))

        return out

    def get_agent_cell_from_id(self, agent:Agent):
        # Get cell type for agent from ID
        match agent.id:
            case 1:
                return Tile(CellType.AGENT_1)
            case 2:
                return Tile(CellType.AGENT_2)
            case 3:
                return Tile(CellType.AGENT_3)
            case 4:
                return Tile(CellType.AGENT_4)


    # Load agent with specified abilities
    # let's say we let agent go on floor, grass, and water
    def load_agent(self, abilities=[1, 3, 4]):
        # NOTE: as of now, abilities MUST contain '1' for normal floors
        # agent = {'id': len(self.agents)+1, 'abilities':abilities, 'position': None}

        # TODO: change this so all agents have ability '1' (FLOOR) by default
        agent = Agent(id=len(self.agents)+1, abilities=abilities, position=None)
        self.agents.append(agent)

    # Place agents in top left-most open blocks
    def place_agents_at_start(self):
        if self.map is None:
            print("No map generated!")
            return False
        
        if not self.agents:
            print("No agents to place!")
            return False
        
        cur_agent = 0

        for row in range(self.rows):
            for col in range(self.cols):
                # print("row: {}, col: {}".format(row, col), self.map[row, col])
                if self.map[row, col].get_type() == CellType.FLOOR:
                    self.agents[cur_agent].position = (row, col)
                    self.state_map[row, col] = self.get_agent_cell_from_id(self.agents[cur_agent])
                    self.state_map[row, col].observe()
                    cur_agent += 1
                    if cur_agent >= len(self.agents):
                        return

    # Load map from file
    def load_map(self):
        pass

    def move_agent(self, agent: Agent, dir):
        # Define move initially as None to handle unexpected cases
        move = None

        if dir == Action.WEST:
            move = (0, -1)
        elif dir == Action.EAST:
            move = (0, 1)
        elif dir == Action.NORTH:
            move = (-1, 0)
        elif dir == Action.SOUTH:
            move = (1, 0)

        # ensure 'move' is not None before proceeding
        if move is not None:
            # print(agent.position, move)
            n_row, n_col = agent.position[0] + move[0], agent.position[1] + move[1]

            # Correctly call and use the result of check_agent_ability
            if self.check_agent_ability(agent, dir, n_row, n_col):
                self.state_map[agent.position] = self.map[agent.position]
                agent.position = (n_row, n_col)
                self.state_map[agent.position] = self.get_agent_cell_from_id(agent)
                # self.state_map[agent.position].observe()
        # else:
            # print(f"Invalid direction {dir}")


    # Move agent in specified direction, if valid
    def check_agent_ability(self, agent:Agent, dir, n_row, n_col):
        if n_row < 0 or n_row >= len(self.map) or n_col < 0 or n_col >= len(self.map[0]):
            # Position is out of bounds
            return False

        match self.map[n_row, n_col].get_type():
            case CellType.OOB:
                # Agent can't move out of bounds
                return False
            case CellType.WALL:
                # Agent can't move into walls
                return False
            case CellType.GRASS:
                if 3 not in agent.abilities:
                    # Agent can't move into grass
                    return False
            case CellType.WATER:
                if 4 not in agent.abilities:
                    # Agent can't move into water
                    return False
                
            # we don't check floor because it's the default walkable cell type

        # Move agent to new position if not already occupied
        for check_agent in self.agents:
            if check_agent.position == (n_row, n_col):
                # print("Position {} already occupied by agent {}".format(check_agent.position, check_agent.id))
                return False
                
        return True

    def get_agent_ability(self, agent:Agent):
        return agent.abilities

    def agent_peek(self, agent:Agent, dir): 
        if agent.position is None:
            # Optionally, print a warning or log this event for debugging
            print(f"Warning: Attempted to peek with agent {agent} having None position.")
            # Return an empty observation or handle this scenario as needed
            return []
    
        # Define the sight range of the agent
        sight_range = agent.rangeOfSight

        # Get the agent's current position
        agent_row, agent_col = agent.position

        observation = []
        # if dir == Action.NORTH:
        #     min_row = max(0, agent_row - sight_range)
        #     for row in range(min_row, agent_row):
        #        observation.append(self.map[row][agent_col])
        #     observation.reverse() # we reverse the list to make it from near to far since we are going from bottom to top
        # elif dir == Action.SOUTH:
        #     max_row = min(len(self.map) - 1, agent_row + sight_range)
        #     for row in range(agent_row + 1, max_row + 1):
        #         observation.append(self.map[row][agent_col])
        # elif dir == Action.EAST:
        #     max_col = min(len(self.map[0]) - 1, agent_col + sight_range)
        #     for col in range(agent_col + 1, max_col + 1):
        #         observation.append(self.map[agent_row][col])
        # elif dir == Action.WEST:
        #     min_col = max(0, agent_col - sight_range)
        #     for col in range(min_col, agent_col):
        #         observation.append(self.map[agent_row][col])
        #     observation.reverse()

        if dir == Action.NORTH:
            for d_row in range(1, sight_range+1):
                n_row = agent_row - d_row
                if n_row < 0 or n_row >= self.rows:
                    obs_tile = Tile(CellType.OOB)
                else:
                    obs_tile = self.map[n_row, agent_col]
                    self.obs_map[n_row, agent_col] = obs_tile
                observation.append({'position':(n_row, agent_col), 'type':obs_tile.get_type()})
        elif dir == Action.SOUTH:
            for d_row in range(1, sight_range+1):
                n_row = agent_row + d_row
                if n_row < 0 or n_row >= self.rows:
                    obs_tile = Tile(CellType.OOB)
                else:
                    obs_tile = self.map[n_row, agent_col]
                    self.obs_map[n_row, agent_col] = obs_tile
                observation.append({'position':(n_row, agent_col), 'type':obs_tile.get_type()})
        elif dir == Action.EAST:
            for d_col in range(1, sight_range+1):
                n_col = agent_col + d_col
                if n_col < 0 or n_col >= self.cols:
                    obs_tile = Tile(CellType.OOB)
                else:
                    obs_tile = self.map[agent_row, n_col]
                    self.obs_map[agent_row, n_col] = obs_tile
                observation.append({'position':(agent_row, n_col), 'type':obs_tile.get_type()})
        elif dir == Action.WEST:
            for d_col in range(1, sight_range+1):
                n_col = agent_col - d_col
                if n_col < 0 or n_col >= self.cols:
                    obs_tile = Tile(CellType.OOB)
                else:
                    obs_tile = self.map[agent_row, n_col]
                    self.obs_map[agent_row, n_col] = obs_tile
                observation.append({'position':(agent_row, n_col), 'type':obs_tile.get_type()})


        # for obs in observation:
        #     self.obs_map[obs['position'][0], obs['position'][1]]

        return observation

    def get_env_state(self):
        pass

    def get_agent_observation(self, agent):
        # Returns the state (types) of blocks in a circular radius around the agents?
        pass

    def get_all_agent_observation(self):
        # Returns the state (types) of blocks in a circular radius around the agents?
        # for agent in self.agents:
        #     pass
        pass

    def calc_agent_observation(self, agent:Agent):
        # Calculate observation for a specific agent

        # TODO: add logic to prevent looking through walls

        for d_row in range(agent.rangeOfSight+1):
            for d_col in range(agent.rangeOfSight+1):
                if d_row == 0 and d_col == 0:
                    # Agent's new position, observation already seen
                    n_row, n_col = agent.position[0] + d_row, agent.position[1] + d_col
                    self.obs_map[n_row, n_col] = self.map[n_row, n_col]
                    continue
                
                # Calc +row, +col
                # print("Agent position: {}".format(agent.position) , "d_row: {}, d_col: {}".format(d_row, d_col))
                n_row, n_col = agent.position[0] + d_row, agent.position[1] + d_col

                # Check if position is in bounds, otherwise mark OOB
                if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                    if self.obs_map[n_row, n_col].get_type() == CellType.UNKNOWN:
                         # mark it as observed
                        self.map[n_row, n_col].observe()
                        self.obs_map[n_row, n_col] = self.map[n_row, n_col]
                        self.cur_step_reward += 1
                else:
                    # OOB case
                    pass

                # If col change not 0, calc +row, -col
                if d_col != 0:
                    n_row, n_col = agent.position[0] + d_row, agent.position[1] - d_col

                    if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                        if self.obs_map[n_row, n_col].get_type() == CellType.UNKNOWN:
                             # mark it as observed
                            self.map[n_row, n_col].observe()
                            self.obs_map[n_row, n_col] = self.map[n_row, n_col]
                            self.cur_step_reward += 1
                    else:
                        # OOB case
                        pass

                # If row change not 0, calc -row, +col
                if d_row != 0:
                    n_row, n_col = agent.position[0] - d_row, agent.position[1] + d_col

                    if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                        if self.obs_map[n_row, n_col].get_type() == CellType.UNKNOWN:
                            # mark it as observed
                            self.map[n_row, n_col].observe()
                            self.obs_map[n_row, n_col] = self.map[n_row, n_col]
                            self.cur_step_reward += 1
                    else:
                        # OOB case
                        pass

                    # If col change not 0, calc -row, -col
                    if d_col != 0:
                        n_row, n_col = agent.position[0] - d_row, agent.position[1] - d_col

                        if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                            if self.obs_map[n_row, n_col].get_type() == CellType.UNKNOWN:
                                # mark it as observed
                                self.map[n_row, n_col].observe()
                                self.obs_map[n_row, n_col] = self.map[n_row, n_col]
                                self.cur_step_reward += 1
                        else:
                            # OOB case
                            pass

        tmp_obs_map = copy.deepcopy(self.obs_map)

        # Set agent position
        tmp_obs_map[agent.position[0], agent.position[1]] = self.get_agent_cell_from_id(agent)
        observation_array = self.map_to_numeric(tmp_obs_map)
        torch.tensor(observation_array)
        
        return tmp_obs_map
    
    def map_to_numeric(self, map_data):
        return np.array([[tile.get_type().value for tile in row] for row in map_data], dtype=np.int32)


    # def convert_to_numpy(self, tmp_obs_map):
    #     # Extracting cell_type.value from each Tile in tmp_obs_map
    #     map_array = [[tile.get_type().value for tile in row] for row in tmp_obs_map]
    #     return np.array(map_array, dtype=np.int32)

    def clear_state_map(self):
        # Clear the state map
        self.state_map = copy.deepcopy(self.map)


    def clear_obs_map(self):
        # Clear the observed map
        self.obs_map[:] = Tile(CellType.UNKNOWN)


    def generate_map(self):
        print("Generating map")
        map = self.initialize_base_terrain()
        self.map = map
        self.apply_cellular_automata()
        self.add_clustered_features(CellType.GRASS, 3, 20)  # 3 clusters, each with 20 cells
        self.add_clustered_features(CellType.WATER, 2, 15)  # 2 clusters, each with 15 cells
        self.ensure_connectivity()

        self.map = np.array(self.map)

        print("Map generation complete")
        return self.map
    
    # Randomly assign base terrain types (floor or wall) to each cell
    def initialize_base_terrain(self):
        new_map = [[Tile(CellType.FLOOR) for _ in range(self.cols)] for _ in range(self.rows)]
        for y in range(self.rows):
            for x in range(self.cols):
                new_map[y][x] = random.choice([Tile(CellType.FLOOR), Tile(CellType.WALL)])
        return new_map

    # Smooth and cluster terrain using cellular automata rules  
    def apply_cellular_automata(self):
        for iteration in range(5):
            # This map will store the results of applying the cellular automata rules for the current iteration
            new_map = [[Tile(CellType.WALL) for _ in range(self.cols)] for _ in range(self.rows)]
            for y in range(self.rows):
                for x in range(self.cols):
                    # This line calculates the number of neighboring cells that are floors (CellType.FLOOR). 
                    # It uses a generator expression within the sum() function to add 1 for each neighboring cell that is a floor.
                    # dx and dy are used to check all adjacent cells (including diagonals) around the current cell (x, y). 
                    floor_neighbors = sum(
                        1
                        for dy in range(-1, 2)
                        for dx in range(-1, 2)
                         # ensures that the indices are within the map bounds.
                        if 0 <= x + dx < self.cols and 0 <= y + dy < self.rows

                         # checks if the neighboring cell is a floor. The sum of these checks gives the total count of floor neighbors.
                        and self.map[y + dy][x + dx].get_type() == CellType.FLOOR
                    )
                    
                    if floor_neighbors >= 5 or self.map[y][x].get_type() == CellType.FLOOR and floor_neighbors >= 4:
                        new_map[y][x]= Tile(CellType.FLOOR)
            self.map = new_map

        # TODO: add a check somewhere around here so there are no sections connected only by diagonal cells (i.e. sections that can't be reached)

    # Add feature features in clusters    
    def add_clustered_features(self, feature_type, clusters, size):
        for _ in range(clusters):
            start_x, start_y = random.randint(0, self.cols - 1), random.randint(0, self.rows - 1)
            for _ in range(size):
                if 0 <= start_x < self.cols and 0 <= start_y < self.rows:
                    self.map[start_y][start_x] = Tile(feature_type)
                # Randomly move the "cluster seed" to simulate natural spreading
                start_x += random.choice([-1, 0, 1])
                start_y += random.choice([-1, 0, 1])

    def ensure_connectivity(self):
        # Flood fill to label each isolated region with a unique ID
        regions = {}  # Maps each cell to its region ID
        region_id = 1
        for y in range(self.rows):
            for x in range(self.cols):
                if self.map[y][x].get_type() == CellType.FLOOR and (x, y) not in regions:
                    # Use flood fill to find and label all cells in this region
                    self.flood_fill(self.map, x, y, Tile(CellType.FLOOR), region_id, regions)
                    region_id += 1

        # If there's only one region, or none, no need to connect anything
        if region_id <= 2:
            return

        # Find edge cells for each region to potentially connect
        edge_cells = self.find_edge_cells(regions, region_id)

        # Connect regions. Here we simply connect each region to the next, but you might use more sophisticated logic
        for i in range(1, region_id - 1):
            self.connect_regions(edge_cells[i], edge_cells[i + 1])

    def flood_fill(self, map, x, y, target_type, region_id, regions):
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows or map[y][x].get_type() != target_type or (x, y) in regions:
            return
        regions[(x, y)] = Tile(region_id)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            self.flood_fill(map, x + dx, y + dy, target_type, region_id, regions)

    def find_edge_cells(self, regions, region_id):
        edge_cells = {rid: [] for rid in range(1, region_id)}
        for (x, y), rid in regions.items():
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.map[ny][nx].cell_type == CellType.WALL or (nx, ny) not in regions:
                        edge_cells[rid].append((x, y))
                        break
                else:
                    edge_cells[rid].append((x, y))
                    break
            # Debug print to check if edge cells are being populated
            print(f"Region {rid}, Edge Cells: {edge_cells[rid]}")
        return edge_cells


    def connect_regions(self, region1, region2):
        if not region1 or not region2:
            # print("One or both regions have no edge cells to connect.")
            return  # Exit if there's nothing to connect
        start = region1[0]
        end = region2[0]
        self.carve_path(start, end)

    def carve_path(self, start, end):
        # Simplified path carving from start to end
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            # Convert the cell to a floor to create a path
            if 0 <= x0 < self.cols and 0 <= y0 < self.rows:
                self.map[y0][x0] = Tile(CellType.FLOOR)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    # Save map to .csv file
    def save_map(self, fname):
        tmp_map = self.map_to_numeric(self.map)
        # print(map)

        tmp_map = np.array(tmp_map, dtype=np.int8)
        np.savetxt(fname, tmp_map, delimiter=',', fmt='%d')

    # Load map from .csv file
    def load_map(self, fname):
        tmp_map = np.genfromtxt(fname, delimiter=',')
        tile_map = [[Tile(CellType(tmp_map[row, col])) for col in range(tmp_map.shape[1])] for row in range(tmp_map.shape[0])]

        # for row in tmp_map:
        #     for tile in row:
        #         tmp_map = Tile(CellType(tile))


        self.map = np.array(tile_map)
        return self.map

    # Get base map data
    def get_map(self):
        return self.map
    
    def get_obs_map(self):
        return self.obs_map

    # Get agent data
    def get_agents(self):
        return self.agents
    
    def is_done(self):
        total_tiles = self.rows * self.cols
        return self.discovered_tiles_num >= total_tiles

    def evaluate_map(self):
        # Evaluate the quality of the map produced by the agents
        pass 




