"""
2048 Player.

Written July 31st, 2017.

The goal of this program is to evolve a bot that can play 2048.

The board is represented by a 4x4 array of numbers:
    0 represents an empty space.
    1 represents the highest-valued piece attained thus far.
    Numbers between 0 and 1 represent the scaled log of the piece's value.

For example, a board arranged like this:
    -   -   2   2
    -   -   2   4
    2   2   4   16
    8   32  128 32

would be represented like this:
    0   0   1   1
    0   0   1   2
    1   1   2   4
    3   5   7   5

and scaled like this:
    0       0     1/7     1/7
    0       0     1/7     2/7
    1/7     1/7   2/7     4/7
    3/7     5/7   1       5/7

Upon a move, the program goes through each row or column:
    First, the row is sorted, with all non-zero elements considered identical.
    Second, each pair of identical elements are combined, the empty space replaced with a zero, and the score incremented.
    Third, the first step is repeated once (and only once).
After each row or column has been scanned, the high score is increased (if necessary).
Finally, a 2 or 4 is spawned randomly

The fitness function is the score of the game, which is incremented by the value of any newly created block.

This would likely be a LOT faster using numpy arrays rather than python lists of lists.
"""

from __future__ import print_function
import os
import neat
import visualize

import math
import random


# Note: This doesn't actually work yet.  I haven't finished debugging it.
def next_move(board, highValue, score, move):
    if move == 1:
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1)
            for i in range(3):
                if row[i] == row[i+1]:
                    row[i] += 1. / math.log(2,highValue)
                    row[i+1] = 0
                    score += round(2**(row[i]*math.log(2,highValue)))
            row.sort(key = lambda x: 0 if x==0 else 1)
    elif move == 2:
        board = map(list, zip(*board))
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1)
            for i in range(3):
                if row[i] == row[i+1]:
                    row[i] += 1. / math.log(2,highValue)
                    row[i+1] = 0
                    score += round(2**(row[i]*math.log(2,highValue)))
            row.sort(key = lambda x: 0 if x==0 else 1)
        board = map(list, zip(*board))
    elif move == 3:
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
            for i in range(4,1,-1):
                if row[i] == row[i-1]:
                    row[i-1] += 1. / math.log(2,highValue)
                    row[i] = 0
                    score += round(2**(row[i-1]*math.log(2,highValue)))
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
    elif move == 4:
        board = map(list, zip(*board))
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
            for i in range(4,1,-1):
                if row[i] == row[i-1]:
                    row[i-1] += 1. / math.log(2,highValue)
                    row[i] = 0
                    score += round(2**(row[i-1]*math.log(2,highValue)))
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
        board = map(list, zip(*board))
    if max([max(row) for row in board]) > 1:
        board = [[elem*(math.log(2,highValue)/(math.log(2,highValue) + 1)) for elem in row] for row in board]
        highValue = 2 * highValue
    zeroPositions = []
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                zeroPositions.append([i,j])
    zeroPositions = random.choice(zeroPositions)
    # In 2048, there is a 90% chance of a new tile being 2, and 10% chance of being 4.
    board[zeroPositions[0]][zeroPositions[1]] = (2 if random.random() < 0.9 else 4)
    return board, highValue, score

next_move(oldBoard, 128, 1000, 1)

tictactoe_inputs = [tuple([random.choice([-1,0,1]) for i in range(9)]) for j in range(25)]
tictactoe_outputs = [[int(i==0) for i in board] for board in tictactoe_inputs]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        numberOfBoards = float(len(tictactoe_inputs))
        genome.fitness = 1.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        #fitnessBoard = [[0 for j in range(len(tictactoe_inputs))] for i in range(9)]

        for xi, xo in zip(tictactoe_inputs, tictactoe_outputs):
            output = net.activate(xi)

            for i in range(9):
                genome.fitness -= float((xo[i] - output[i]) ** 2) / (9.0 * numberOfBoards)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
     neat.DefaultSpeciesSet, neat.DefaultStagnation,
     config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000-1))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 5000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(tictactoe_inputs, tictactoe_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'i 1,1', -2: 'i 1,2', -3: 'i 1,3', -4: 'i 2,1', -5: 'i 2,2', -6: 'i 2,3', -7: 'i 3,1', -8: 'i 3,2', -9: 'i 3,3', 0: 'o 1,1', 1: 'o 1,2', 2: 'o 1,3', 3: 'o 2,1', 4: 'o 2,2', 5: 'o 2,3', 6: 'o 3,1', 7: 'o 3,2', 8: 'o 3,3'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-tictactoe')
    run(config_path)
