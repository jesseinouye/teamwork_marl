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
    OOB = 3         # Out Of Bounds (OOB) cell


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
    def load_agent(self, abilities=[1]):
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

    # Move agent in specified direction, if valid
    def move_agent(self, agent:Agent, dir):
        # Check validity from least to most computationally taxing
        #   Check movement within map boundary
        #   Check collision with walls
        #   Check collision with other agents
        if dir == Action.WEST:
            move = (0, -1)
        if dir == Action.EAST:
            move = (0, 1)
        if dir == Action.NORTH:
            move = (-1, 0)
        if dir == Action.SOUTH:
            move = (1, 0)

        n_row, n_col = agent.position[0] + move[0], agent.position[1] + move[1]

        # if map[n_row][n_col] == CellType.OOB:
        #     # Agent can't move out of bounds
        #     return
        
        match self.map[n_row][n_col]:
            case CellType.OOB:
                # Agent can't move out of bounds
                return
            case CellType.WALL:
                # Agent can't move into walls
                return
            case CellType.FLOOR:
                # Move agent to new position if not already occupied
                for check_agent in self.agents:
                    if check_agent.position == (n_row, n_col):
                        print("Position {} already occupied by agent {}".format((check_agent.position, check_agent.id)))
                        return
                
                agent.position = (n_row, n_col)


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

        for d_row in range(self.agent_obs_dist+1):
            for d_col in range(self.agent_obs_dist+1):
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
        # dirs = copy.deepcopy(dirs)
        random.shuffle(dirs)
        prev_dir = random.randint(0, 3)

        # weights_base = [0.2, 0.2, 0.2, 0.2]
        weights_base = [0.1, 0.1, 0.1, 0.1]

        map = [[CellType.WALL for _ in range(self.cols)] for _ in range(self.rows)]

        current_pos = (random.randint(0, self.cols-1), random.randint(0, self.rows-1))

        all_dirs = []

        floor_count = 1
        while floor_count < self.rows * self.cols * 0.60:
            x, y = current_pos

            weights = copy.deepcopy(weights_base)
            # weights[prev_dir] = 0.4
            weights[prev_dir] = 0.7 # heavily prioritizes straight lines

            # print("weights: {}\npop: {}".format(weights, dirs_idx))

            new_dir_idx = random.choices(dirs_idx, weights)[0]
            dx, dy = dirs[new_dir_idx]

            all_dirs.append(new_dir_idx)

            nx, ny = x + dx, y + dy

            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                if map[nx][ny] == CellType.WALL:
                    floor_count += 1

                map[nx][ny] = CellType.FLOOR
                current_pos = (nx, ny)
                
            prev_dir = new_dir_idx

        print("dir counts: {}".format(Counter(all_dirs)))

            # random.shuffle(dirs)
            # moved = False

            # for dx, dy in dirs:
            #     nx, ny = x + dx, y + dy

            #     if 0 <= nx < self.cols and 0 <= ny < self.rows:
            #         # print("new pos: {}".format((nx, ny)))
                    
            #         if map[nx][ny] == 0:
            #             floor_count += 1

            #         map[nx][ny] = 1

            #         current_pos = (nx, ny)
            #         moved = True
            #         break
        
        self.map = map
        return map
    
    # Get base map data
    def get_map(self):
        return self.map
    
    # Get agent data
    def get_agents(self):
        return self.agents





