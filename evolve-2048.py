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

oldBoard = [[0,0,1./7,1./7],[0,0,1./7,2./7],[1./7,1./7,2./7,4./7],[3./7,5./7,7./7,5./7]]

Upon a move, the program goes through each row or column:
    First, the row is sorted, with all non-zero elements considered identical.
    Second, each pair of identical elements are combined, the empty space replaced with a zero, and the score incremented.
    Third, the first step is repeated once (and only once).
After each row or column has been scanned, the high score is increased (if necessary).
Finally, a 2 or 4 is spawned randomly in one of the positions currently occupied by 0.
    A 4 has a 10% chance of spawning, while a 2 has a 90% chance.
The fitness function is the score of the game, which is incremented by the value of any newly created block.

This would likely be a LOT faster using numpy arrays rather than python lists of lists.
"""

from __future__ import print_function
import os
import neat
import visualize

import math
import random
import copy
import time


# Note: This doesn't actually work yet.  I haven't finished debugging it.
def next_move(oldBoard, highValue, score, move):
    board = copy.deepcopy(oldBoard)
    if move == 0:
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1)
            for i in range(3):
                if row[i] == row[i+1] and row[i] != 0:
                    row[i] += (1. / math.log(highValue,2))
                    row[i+1] = 0
                    score += round(2**(1.+row[i]*math.log(highValue,2)))
            row.sort(key = lambda x: 0 if x==0 else 1)
    elif move == 1:
        board = map(list, zip(*board))
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1)
            for i in range(3):
                if row[i] == row[i+1] and row[i] != 0:
                    row[i] += (1. / math.log(highValue,2))
                    row[i+1] = 0
                    score += round(2**(1.+row[i]*math.log(highValue,2)))
            row.sort(key = lambda x: 0 if x==0 else 1)
        board = map(list, zip(*board))
    elif move == 2:
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
            for i in range(3,0,-1):
                if row[i] == row[i-1] and row[i] != 0:
                    row[i-1] += (1. / math.log(highValue,2))
                    row[i] = 0
                    score += round(2**(1.+row[i]*math.log(highValue,2)))
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
    elif move == 3:
        board = map(list, zip(*board))
        for row in board:
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
            for i in range(3,0,-1):
                if row[i] == row[i-1] and row[i] != 0:
                    row[i-1] += (1. / math.log(highValue,2))
                    row[i] = 0
                    score += round(2**(1.+row[i]*math.log(highValue,2)))
            row.sort(key = lambda x: 0 if x==0 else 1, reverse=True)
        board = map(list, zip(*board))
    if max([max(row) for row in board]) > 1:
        board = [[elem*(math.log(highValue,2)/(math.log(highValue,2) + 1)) for elem in row] for row in board]
        highValue = 2 * highValue
    zeroPositions = []
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                zeroPositions.append([i,j])
    try:
        zeroPositions = random.choice(zeroPositions)
        board[zeroPositions[0]][zeroPositions[1]] = ((1. / math.log(highValue,2)) if random.random() < 0.9 else (2. / math.log(highValue,2)))
    except IndexError:
        pass
    return board, highValue, score

#oldBoard = [[0,0,1./7,1./7],[0,0,1./7,2./7],[1./7,1./7,2./7,4./7],[3./7,5./7,7./7,5./7]]
#next_move(oldBoard, 128, 1000, 3)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        numGames = 10
        for game in range(numGames):
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            gameOver = False
            currentBoard = [[0,0,0,0],[0,0.5,0,0],[0,0,1,0],[0,0,0,0]]
            h = 4
            while not gameOver:
                moveMatrix = net.activate([currentBoard[i][j] for i in range(4) for j in range(4)])
                gameOver = True # This halts the loop if the for loop returns False.
                for i in sorted(range(4), key=lambda k: moveMatrix[k], reverse = True):
                    oldScore = genome.fitness
                    newBoard, newHighValue, newScore = next_move(currentBoard, h, oldScore, i)
                    if False in [currentBoard[i][j]==newBoard[i][j] for i in range(4) for j in range(4)]:
                        currentBoard = copy.deepcopy(newBoard)
                        h = newHighValue
                        genome.fitness = newScore
                        gameOver = False
                        #[print([int(2**(i*math.log(h,2))) if i!=0 else 0 for i in row]) for row in currentBoard]
                        #print("")
                        #time.sleep(1)
                        break
        genome.fitness = genome.fitness / numGames

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
    winner = p.run(eval_genomes, 10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    oldScore = 0.0
    gameOver = False
    currentBoard = [[0,0,0,0],[0,2,0,0],[0,0,4,0],[0,0,0,0]]
    h = 4
    while not gameOver:
        moveMatrix = winner_net.activate([currentBoard[i][j] for i in range(4) for j in range(4)])
        gameOver = True
        for i in sorted(range(4), key=lambda k: moveMatrix[k], reverse = True):
            newBoard, newHighValue, newScore = next_move(currentBoard, h, oldScore, i)
            if False in [currentBoard[i][j]==newBoard[i][j] for i in range(4) for j in range(4)]:
                currentBoard = copy.deepcopy(newBoard)
                h = newHighValue
                oldScore = newScore
                gameOver = False
                break
    print("Winning Score: {!r}, highest-valued tile {!r}".format(oldScore, h))
    print("Final board state: ")
    [print([2**(i*math.log(h,2)) if i!=0 else 0 for i in row]) for row in currentBoard]

    node_names = {
        -1: '1,1', -2: '1,2', -3: '1,3', -4: '1,4',
        -5: '2,1', -6: '2,2', -7: '2,3', -8: '2,4',
        -9: '3,1', -10: '3,2', -11: '3,3', -12: '3,4',
        -13: '4,1', -14: '4,2', -15: '4,3', -16: '4,4',
        0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up'
        }
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
    config_path = os.path.join(local_dir, 'config-2048')
    run(config_path)
