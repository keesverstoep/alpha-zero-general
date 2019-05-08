from Coach import Coach
from awari.AwariGame import AwariGame
from awari.keras.NNet import NNetWrapper as nn
# from awari.tensorflow.NNet import NNetWrapper as nn
from utils import *
import numpy as np

args = dotdict({
    # 'numIters': 10,
    # 'numIters': 20,
    'numIters': 1,
    # 'numIters': 150,
    # 'numEps': 100,
    # 'numEps': 25,
    'numEps': 5,
    # 'tempThreshold': 15,
    # increased so choices are not stuck too soon:
    # 'tempThreshold': 60,
    # TODO: further increased, need to be beyond regular draw limit?
    'tempThreshold': 400,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    # 'numMCTSSims': 10,
    # 'numMCTSSims': 25,
    'numMCTSSims': 100,
    # 'arenaCompare': 20,
    'arenaCompare': 10,
    'cpuct': 1,

    'checkpoint': './temp/awari-keras/',

    # 'load_model': False,
    'load_model': True,
    'load_folder_file': ('./temp/awari-keras/', 'checkpoint_snapshot.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    g = AwariGame()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    # c.learn()
    c.loadTrainExamples()
    num = 0
    for i in range(len(c.trainExamplesHistory)):
        for elem in c.trainExamplesHistory[i]:
            board, policy, val = elem
            if np.array_equal(board, [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0]]):
                print('new game: ' + str(num))
                num += 1
                move = 0
            strboard = ""
            n = 0
            for pit in board[0]:
                strboard += "%2d " % pit
                n += 1
                if n % 6 == 0:
                    strboard += " "
            print(("%3d " % move) + ": " + strboard + (": %3d" % val) + ": " + str(policy))
            move += 1

