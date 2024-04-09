import pygame
from environment_engine import EnvEngine, Action, Agent, CellType

WIDTH = 800
HEIGHT = 800
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

class Visualizer():
    def __init__(self) -> None:
        self.env = EnvEngine()

        self.rows = self.env.rows
        self.cols = self.env.cols

        self.cell_size = WIDTH // self.cols

        self.map = self.env.generate_map_random_walk()
        self.env.load_agent(abilities=[1])
        # self.env.load_agent(abilities=[1,2])
        # self.env.place_agents()

        pygame.init()

    def draw_map(self, screen, map, agents:list[Agent]=[]):
        for row in range(self.rows):
            for col in range(self.cols):
                if map[row][col] == CellType.WALL:
                    pygame.draw.rect(screen, BLACK, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(screen, WHITE, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

        if agents:
            for agent in agents:
                if agent.position is not None:
                    pygame.draw.rect(screen, RED, (agent.position[1] * self.cell_size, agent.position[0] * self.cell_size, self.cell_size, self.cell_size))

    def main(self):
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("teamwork")
        screen.fill(WHITE)

        running = True
        i = 0
        while running:
            # print("running iter: {}".format(i))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    print("got keydown")
                    if event.key == pygame.K_LEFT:
                        self.env.move_agent(agents[0], Action.WEST)
                    if event.key == pygame.K_RIGHT:
                        self.env.move_agent(agents[0], Action.EAST)
                    if event.key == pygame.K_UP:
                        self.env.move_agent(agents[0], Action.NORTH)
                    if event.key == pygame.K_DOWN:
                        self.env.move_agent(agents[0], Action.SOUTH)
                    
                    if event.key == pygame.K_SPACE:
                        print("got space")
                        self.env.place_agents()
                        obs = self.env.calc_agent_observation(agents[0])
                        print("obs: {}".format(obs))

            map = self.env.get_map()
            agents = self.env.get_agents()

            self.draw_map(screen, map, agents)
            pygame.display.update()
            # pygame.display.flip()
            # i += 1

if __name__ == "__main__":
    vis = Visualizer()
    vis.main()

            
