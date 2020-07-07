import pygame
import neat
import random
import sys
import pickle

win = pygame.display.set_mode((800, 800))
pygame.display.set_caption("neat-car")

carImage = pygame.image.load("car.png")

carRedImage = pygame.image.load("carRed.png")
carOrangeImage = pygame.image.load("carOrange.png")
carBlueImage = pygame.image.load("carBlue.png")
carYellowImage = pygame.image.load("carYellow.png")
carGreenImage = pygame.image.load("carGreen.png")

roadImage = pygame.image.load("road.png")


class CarObstacle:
    def __init__(self, x, image):
        self.x = x
        self.y = -250
        self.image = image
        self.speed = 18

    def move(self):
        self.y += self.speed

    def draw(self):
        win.blit(self.image, (self.x, self.y))

    def collide(self, car):
        player_mask = car.get_mask()
        o_mask = pygame.mask.from_surface(self.image)
        o_offset = (round(self.x) - round(car.x), round(self.y) - round(car.y))

        o_point = player_mask.overlap(o_mask, o_offset)

        if o_point:
            return True

        return False


class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dir):
        if self.x != 45:
            if dir == "left":
                self.x -= 200
        if self.x != 645:
            if dir == "right":
                self.x += 200

    def draw(self):
        win.blit(carImage, (self.x, self.y))

    def get_mask(self):
        return pygame.mask.from_surface(carImage)


def gameLoop(genomes, config):
    nets = []
    ge = []
    cars = []

    score = 0

    positions = [45, 245, 445, 645]
    colors = [carBlueImage, carGreenImage, carRedImage, carOrangeImage, carYellowImage]
    carObstacle = CarObstacle(random.choice(positions), random.choice(colors))

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(45, 520))
        g.fitness = 0
        ge.append(g)

    while True:
        pygame.time.Clock().tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        for x, car in enumerate(cars):
            ge[x].fitness += 0.1

            output = nets[cars.index(car)].activate(
                (car.y, car.x, carObstacle.x, carObstacle.y, abs(car.y - carObstacle.y), abs(car.x - carObstacle.x)))
            if output[0] > 0.5:
                for genome in ge:
                    genome.fitness -= 0.1
                car.move("right")
            if output[1] > 0.5:
                for genome in ge:
                    genome.fitness -= 0.1
                car.move("left")

        win.blit(roadImage, (0, 0))

        if carObstacle.y >= 945:
            carObstacle.speed += 0.30
            carObstacle.x = random.choice(positions)
            carObstacle.image = random.choice(colors)
            carObstacle.y = -250
            score += 1
            for genome in ge:
                genome.fitness += 5

        carObstacle.move()

        carObstacle.draw()
        for car in cars:
            car.draw()

        for car in cars:
            if carObstacle.collide(car):
                ge[cars.index(car)].fitness -= 1
                nets.pop(cars.index(car))
                ge.pop(cars.index(car))
                cars.pop(cars.index(car))

        if len(cars) <= 0:
            break

        print(score)

        if score >= 40:
            for genome in ge:
                genome.fitness += 10
            break

        pygame.display.update()


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(gameLoop, 10)

    pickle.dump(winner, open('winner.pkl', 'wb'))

    print('\nBest genome:\n{!s}'.format(winner))


def runNeuralNetwork():
    import os
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)


if __name__ == '__main__':
    runNeuralNetwork()
