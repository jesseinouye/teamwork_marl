# Types of cells
from enum import IntEnum


class CellType(IntEnum):
    UNKNOWN = 0   # Default
    FLOOR = 1       # Floor cell
    WALL = 2        # Wall cell
    GRASS = 3       # Grass cell
    WATER = 4       # Water cell
    OOB = 5         # Out Of Bounds (OOB) cell
    AGENT_1 = 6     # Agent 1 in this cell
    AGENT_2 = 7     # Agent 2 in this cell
    AGENT_3 = 8     # Agent 3 in this cell
    AGENT_4 = 9    # Agent 4 in this cell
    NULL = 10

    def is_observable(self):
         # Cells that are considered observed if they are known types of terrain
        return self in [CellType.FLOOR, CellType.WALL, CellType.GRASS, CellType.WATER]


class Tile:
    def __init__(self, cell_type=CellType.UNKNOWN):
        self.cell_type = cell_type
        self.observed = False  # Initially, the tile has not been observed.

    def get_type(self):
        return self.cell_type
    
    def observe(self):
        self.observed = True  # Mark the tile as observed

    def is_observed(self):
        return self.observed

    def __str__(self):
        return f"{self.cell_type.name} ({'is currently observed' if self.observed else 'is currently unobserved'})"
    
