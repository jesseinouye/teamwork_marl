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
    UNKNOWN = 6


# Agent observations
# class AgentObs(CellType):
#     UNKNOWN = 6


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
        self.obs_map = None
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
                    obs_type = CellType.OOB
                else:
                    obs_type = self.map[n_row][agent_col]
                    self.obs_map[n_row][agent_col] = obs_type
                observation.append({'position':(n_row, agent_col), 'type':obs_type})
        elif dir == Action.SOUTH:
            for d_row in range(1, sight_range+1):
                n_row = agent_row + d_row
                if n_row < 0 or n_row >= self.rows:
                    obs_type = CellType.OOB
                else:
                    obs_type = self.map[n_row][agent_col]
                    self.obs_map[n_row][agent_col] = obs_type
                observation.append({'position':(n_row, agent_col), 'type':obs_type})
        elif dir == Action.EAST:
            for d_col in range(1, sight_range+1):
                n_col = agent_col + d_col
                if n_col < 0 or n_col >= self.cols:
                    obs_type = CellType.OOB
                else:
                    obs_type = self.map[agent_row][n_col]
                    self.obs_map[agent_row][n_col] = obs_type
                observation.append({'position':(agent_row, n_col), 'type':obs_type})
        elif dir == Action.WEST:
            for d_col in range(1, sight_range+1):
                n_col = agent_col - d_col
                if n_col < 0 or n_col >= self.cols:
                    obs_type = CellType.OOB
                else:
                    obs_type = self.map[agent_row][n_col]
                    self.obs_map[agent_row][n_col] = obs_type
                observation.append({'position':(agent_row, n_col), 'type':obs_type})


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


    # # Generate a map using random walk with weighted directions
    # # (prefers direction it previously went to create hallways)
    # def generate_map_random_walk(self):
    #     print("Generating map")
    #     dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    #     dirs_idx = [0, 1, 2, 3]
    #     random.shuffle(dirs)
    #     prev_dir = random.randint(0, 3)

    #     weights_base = [0.1, 0.1, 0.1, 0.1]

    #     map = [[CellType.WALL for _ in range(self.cols)] for _ in range(self.rows)]

    #     current_pos = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))

    #     all_dirs = []
    #     floor_count = 1
    #     target_floor_count = self.rows * self.cols * 0.60  # Adjust floor target as needed

    #     while floor_count < target_floor_count:
    #         x, y = current_pos
    #         weights = weights_base.copy()
    #         weights[prev_dir] = 0.7  # Heavily prioritizes straight lines

    #         new_dir_idx = random.choices(dirs_idx, weights)[0]
    #         dx, dy = dirs[new_dir_idx]
    #         nx, ny = x + dx, y + dy

    #         if 0 <= nx < self.cols and 0 <= ny < self.rows:
    #             if map[ny][nx] == CellType.WALL:
    #                 floor_count += 1
    #                 map[ny][nx] = CellType.FLOOR  # Mark as floor
    #             current_pos = (nx, ny)
    #         prev_dir = new_dir_idx

    #     # Additional step to introduce GRASS and WATER
       
    #     for _ in range(int(self.rows * self.cols * 0.15)):  # Adjust grass/water target as needed
    #         for cell_type in [CellType.GRASS, CellType.WATER]:
    #             gx, gy = random.randint(0, self.cols - 1), random.randint(0, self.rows - 1)
    #             map[gy][gx] = cell_type

    #     print("dir counts: {}".format(Counter(all_dirs)))
    #     self.map = map
    #     return map

    def generate_map(self):
        print("Generating map")
        map = self.initialize_base_terrain()
        self.map = map
        self.apply_cellular_automata()
        self.add_clustered_features(CellType.GRASS, 3, 20)  # 3 clusters, each with 20 cells
        self.add_clustered_features(CellType.WATER, 2, 15)  # 2 clusters, each with 15 cells
        self.ensure_connectivity()

        # Create observation map
        self.obs_map = [[CellType.UNKNOWN for _ in range(self.cols)] for _ in range(self.rows)]

        print("Map generation complete")
        return self.map
    
    # Randomly assign base terrain types (floor or wall) to each cell
    def initialize_base_terrain(self):
        new_map = [[CellType.FLOOR for _ in range(self.cols)] for _ in range(self.rows)]
        for y in range(self.rows):
            for x in range(self.cols):
                new_map[y][x] = random.choice([CellType.FLOOR, CellType.WALL])
        return new_map

    # Smooth and cluster terrain using cellular automata rules  
    def apply_cellular_automata(self):
        for iteration in range(5):
            # This map will store the results of applying the cellular automata rules for the current iteration
            new_map = [[CellType.WALL for _ in range(self.cols)] for _ in range(self.rows)]
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
                        and self.map[y + dy][x + dx] == CellType.FLOOR
                    )
                    
                    if floor_neighbors >= 5 or self.map[y][x] == CellType.FLOOR and floor_neighbors >= 4:
                        new_map[y][x] = CellType.FLOOR
            self.map = new_map

    # Add feature features in clusters    
    def add_clustered_features(self, feature_type, clusters, size):
        for _ in range(clusters):
            start_x, start_y = random.randint(0, self.cols - 1), random.randint(0, self.rows - 1)
            for _ in range(size):
                if 0 <= start_x < self.cols and 0 <= start_y < self.rows:
                    self.map[start_y][start_x] = feature_type
                # Randomly move the "cluster seed" to simulate natural spreading
                start_x += random.choice([-1, 0, 1])
                start_y += random.choice([-1, 0, 1])

    def ensure_connectivity(self):
        # Flood fill to label each isolated region with a unique ID
        regions = {}  # Maps each cell to its region ID
        region_id = 1
        for y in range(self.rows):
            for x in range(self.cols):
                if self.map[y][x] == CellType.FLOOR and (x, y) not in regions:
                    # Use flood fill to find and label all cells in this region
                    self.flood_fill(self.map, x, y, CellType.FLOOR, region_id, regions)
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
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows or map[y][x] != target_type or (x, y) in regions:
            return
        regions[(x, y)] = region_id
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            self.flood_fill(map, x + dx, y + dy, target_type, region_id, regions)

    def find_edge_cells(self, regions, region_id):
        edge_cells = {rid: [] for rid in range(1, region_id)}
        for (x, y), rid in regions.items():
            # Check each neighboring cell to determine if the current cell is an edge cell
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                # Ensure the neighboring cell is within the map boundaries
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if self.map[ny][nx] == CellType.WALL or (nx, ny) not in regions:
                        edge_cells[rid].append((x, y))
                        break  # No need to check other neighbors once we know this is an edge cell
                else:
                    # The neighboring cell is out of bounds, which also makes this an edge cell
                    edge_cells[rid].append((x, y))
                    break  # No need to check other neighbors
        return edge_cells


    def connect_regions(self, region1, region2):
        # For simplicity, connect the first edge cell of region1 to the first edge cell of region2
        start = region1[0]
        end = region2[0]
        # Implement path carving between start and end. You can use Bresenham's line algorithm or a simple direct path.
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
                self.map[y0][x0] = CellType.FLOOR
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy


    # Get base map data
    def get_map(self):
        return self.map
    
    def get_obs_map(self):
        return self.obs_map

    # Get agent data
    def get_agents(self):
        return self.agents





