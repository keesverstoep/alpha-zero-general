# from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
# from pickle import Pickler, Unpickler
# instead use json for readability and compatibility with other frameworks
import json
from random import shuffle
import sys
# temp hack:
from awari.AwariLogic import Board


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        moves = 0
        max_moves = self.mcts.MAX_TREE_DEPTH
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            # TODO: look carefully into good settings for tempThreshold. Game dependent!
            # Don't want to be stuck in a loop at lower levels.
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b.tolist(), self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            prevboard = board
            prevplayer = self.curPlayer
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)
            moves += 1
            if moves > max_moves:
                r = 1e-4

            if r!=0:
                # also add final move that finished the game
                if 1:  # r == 1 or r == -1 or r == 1e-4:
                    canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
                    b = Board(6)
                    b.pieces = np.copy(canonicalBoard)
                    # also add a (fake but legal) pi for the final move; but don't do MCTS for this
                    moves = b.get_legal_moves(self.curPlayer)
                    if len(moves) > 0:
                        # print('board: ' + str(b.pieces) + ' canonicalBoard: ' + str(canonicalBoard) + ' legal moves: ' + str(moves))
                        pi = [0, 0, 0, 0, 0, 0, 0]
                        for i in moves:
                            pi[i] = 1/len(moves)
                    else:
                        pi = [0, 0, 0, 0, 0, 0, 1]
                    # trainExamples.append([canonicalBoard, self.curPlayer, pi, None])
                    trainExamples.append([canonicalBoard.tolist(), self.curPlayer, pi, None])
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                # iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                # Regular list so it it json serializable
                iterationTrainExamples = []
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
    
                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    iterationTrainExamples += self.executeEpisode()
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examlpes before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        # with open(filename, "wb+") as f:
        #     Pickler(f).dump(self.trainExamplesHistory)
        with open(filename, "w+", encoding="utf8") as f:
            json.dump(self.trainExamplesHistory, f)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            # with open(examplesFile, "rb") as f:
            #     self.trainExamplesHistory = Unpickler(f).load()
            with open(examplesFile, "r", encoding="utf8") as f:
                self.trainExamplesHistory = json.load(f)
                # for export of game info to the other alpha-zero implementations
                if 0:
                   elist = self.trainExamplesHistory[0]
                   print('train: [')
                   for e in elist:
                       board, pi, val = e
                       # print('train: board ' + str(board[0]) + " pi " + str(pi) + " val " + str(val))
                       strboard = ""
                       for i in board[0]:
                           strboard += "%d:" % i
                       print('train: ' + str([str(strboard), pi, val]) + ",")
                   print('train: ]')
                   sys.exit()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
