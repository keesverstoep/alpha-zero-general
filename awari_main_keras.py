import logging

import coloredlogs

from Coach import Coach
from awari.AwariGame import AwariGame
from awari.keras.NNet import NNetWrapper as nn
# from awari.tensorflow.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100,
    # 'numIters': 60,
    # 'numIters': 40,
    # 'numIters': 30,
    # 'numIters': 20,
    # 'numIters': 10,
    # 'numIters': 3,
    # 'numIters': 150,
    # 'numEps': 100,
    # 'numEps': 25,
    'numEps': 20,
    # 'numEps': 10,
    # 'numEps': 5,
    # 'numEps': 3,
    # 'tempThreshold': 15,
    # increased so choices are not stuck too soon:
    # 'tempThreshold': 60,
    # TODO: further increased, need to be beyond regular draw limit?
    'tempThreshold': 400,
    'updateThreshold': 0.54,
    'maxlenOfQueue': 200000,
    # 'numMCTSSims': 10,
    # 'numMCTSSims': 25,
    # 'numMCTSSims': 100,
    'numMCTSSims': 50,
    'arenaCompare': 20,
    # 'arenaCompare': 11,
    # 'arenaCompare': 10,
    'cpuct': 1,

    'checkpoint': './temp/awari-keras/',
    # 'checkpoint': './temp/awari-keras-new/',

    'load_model': False,
    # 'load_model': True,
    'load_folder_file': ('./temp/awari-keras/', 'checkpoint_snapshot.pth.tar'),
    # 'load_folder_file': ('./temp/awari-keras-new/', 'checkpoint_snapshot.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    log.info('Loading %s...', AwariGame.__name__)
    g = AwariGame()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

