# import numpy
# import pygame
import random
import copy
from typing import List
from enum import Enum
from collections import Counter


# Types of cells
class CellType(Enum):
    UNDEFINED = 0   # Default
    FLOOR = 1       # Floor cell
    WALL = 2        # Wall cell
    GRASS = 3       # Grass cell
    WATER = 4       # Water cell
    OOB = 5         # Out Of Bounds (OOB) cell


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
class EnvEngine():
    def __init__(self) -> None:
        self.map_data = []
        
        # Parameters
        self.rows = 25
        self.cols = 25

        self.agent_obs_dist = 3

        self.map = None
        self.agents  : List[Agent] = []

    # Load agent with specified abilities
    # let's say we let agent go on floor, grass, and water
    def load_agent(self, abilities=[1, 3, 4]):
        # NOTE: as of now, abilities MUST contain '1' for normal floors
        # agent = {'id': len(self.agents)+1, 'abilities':abilities, 'position': None}
        agent = Agent(id=len(self.agents)+1, abilities=abilities, position=None)
        self.agents.append(agent)

    # Place agents in top left-most open blocks
    def place_agents(self):
        if self.map is None:
            print("No map generated!")
            return False
        
        if not self.agents:
            print("No agents to place!")
            return False
        
        starting_pos = (0, 0)
        cur_agent = 0

        for row in range(self.rows):
            for col in range(self.cols):
                if self.map[row][col] == CellType.FLOOR:
                    self.agents[cur_agent].position = (row, col)
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
            n_row, n_col = agent.position[0] + move[0], agent.position[1] + move[1]

            # Correctly call and use the result of check_agent_ability
            if self.check_agent_ability(agent, dir, n_row, n_col):
                agent.position = (n_row, n_col)
        else:
            print(f"Invalid direction {dir}")

    # Move agent in specified direction, if valid
    def check_agent_ability(self, agent:Agent, dir, n_row, n_col):
        if n_row < 0 or n_row >= len(self.map) or n_col < 0 or n_col >= len(self.map[0]):
            # Position is out of bounds
            return False

        match self.map[n_row][n_col]:
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
                print("Position {} already occupied by agent {}".format((check_agent.position, check_agent.id)))
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
        if dir == Action.NORTH:
            min_row = max(0, agent_row - sight_range)
            for row in range(min_row, agent_row):
                observation.append(self.map[row][agent_col])
            observation.reverse() # we reverse the list to make it from near to far since we are going from bottom to top
        elif dir == Action.SOUTH:
            max_row = min(len(self.map) - 1, agent_row + sight_range)
            for row in range(agent_row + 1, max_row + 1):
                observation.append(self.map[row][agent_col])
        elif dir == Action.EAST:
            max_col = min(len(self.map[0]) - 1, agent_col + sight_range)
            for col in range(agent_col + 1, max_col + 1):
                observation.append(self.map[agent_row][col])
        elif dir == Action.WEST:
            min_col = max(0, agent_col - sight_range)
            for col in range(min_col, agent_col):
                observation.append(self.map[agent_row][col])
            observation.reverse()

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
        obs = []

        for d_row in range(agent.rangeOfSight+1):
            for d_col in range(agent.rangeOfSight+1):
                if d_row == 0 and d_col == 0:
                    n_row, n_col = agent.position[0] + d_row, agent.position[1] + d_col
                    obs.append({'cell_pos':(n_row, n_col), 'type':self.map[n_row][n_col]})    
                    continue
                
                # Calc +row, +col
                n_row, n_col = agent.position[0] + d_row, agent.position[1] + d_col

                # Check if position is in bounds, otherwise mark OOB
                if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                    obs.append({'cell_pos':(n_row, n_col), 'type':self.map[n_row][n_col]})
                else:
                    obs.append({'cell_pos':(n_row, n_col), 'type':CellType.OOB})

                # # Calc -row, -col
                # n_row, n_col = agent.position[0] - d_row, agent.position[1] - d_col

                # if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                #     obs.append({'cell_pos':(n_row, n_col), 'type':self.map[n_row][n_col]})
                # else:
                #     obs.append({'cell_pos':(n_row, n_col), 'type':CellType.OOB})

                # If col change not 0, calc +row, -col
                if d_col != 0:
                    n_row, n_col = agent.position[0] + d_row, agent.position[1] - d_col

                    if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                        obs.append({'cell_pos':(n_row, n_col), 'type':self.map[n_row][n_col]})
                    else:
                        obs.append({'cell_pos':(n_row, n_col), 'type':CellType.OOB})

                # If row change not 0, calc -row, +col
                if d_row != 0:
                    n_row, n_col = agent.position[0] - d_row, agent.position[1] + d_col

                    if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                        obs.append({'cell_pos':(n_row, n_col), 'type':self.map[n_row][n_col]})
                    else:
                        obs.append({'cell_pos':(n_row, n_col), 'type':CellType.OOB})

                    # If col change not 0, calc -row, -col
                    if d_col != 0:
                        n_row, n_col = agent.position[0] - d_row, agent.position[1] - d_col

                        if 0 <= n_row < self.cols and 0 <= n_col < self.rows:
                            obs.append({'cell_pos':(n_row, n_col), 'type':self.map[n_row][n_col]})
                        else:
                            obs.append({'cell_pos':(n_row, n_col), 'type':CellType.OOB})

        return obs


    # Generate a map using random walk with weighted directions
    # (prefers direction it previously went to create hallways)
    def generate_map_random_walk(self):
        print("Generating map")
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        dirs_idx = [0, 1, 2, 3]
        random.shuffle(dirs)
        prev_dir = random.randint(0, 3)

        weights_base = [0.1, 0.1, 0.1, 0.1]

        map = [[CellType.WALL for _ in range(self.cols)] for _ in range(self.rows)]

        current_pos = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))

        all_dirs = []
        floor_count = 1
        target_floor_count = self.rows * self.cols * 0.60  # Adjust floor target as needed

        while floor_count < target_floor_count:
            x, y = current_pos
            weights = weights_base.copy()
            weights[prev_dir] = 0.7  # Heavily prioritizes straight lines

            new_dir_idx = random.choices(dirs_idx, weights)[0]
            dx, dy = dirs[new_dir_idx]
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                if map[ny][nx] == CellType.WALL:
                    floor_count += 1
                    map[ny][nx] = CellType.FLOOR  # Mark as floor
                current_pos = (nx, ny)
            prev_dir = new_dir_idx

        # Additional step to introduce GRASS and WATER
        for _ in range(int(self.rows * self.cols * 0.10)):  # Adjust grass/water target as needed
            for cell_type in [CellType.GRASS, CellType.WATER]:
                gx, gy = random.randint(0, self.cols - 1), random.randint(0, self.rows - 1)
                if map[gy][gx] == CellType.FLOOR:
                    map[gy][gx] = cell_type

        print("dir counts: {}".format(Counter(all_dirs)))
        self.map = map
        return map

    
    # Get base map data
    def get_map(self):
        return self.map
    
    # Get agent data
    def get_agents(self):
        return self.agents





