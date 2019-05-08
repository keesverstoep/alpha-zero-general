import numpy as np
import random
from math import inf as infinity
from awari.AwariLogic import Board
import random

"""
Random, Greedy, MiniMax and Human-interacting players for
the game of Awari,
Based on the OthelloPlayers by Surag Nair.
"""

class RandomAwariPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        select = np.random.randint(self.game.getActionSize())
        valid = self.game.getValidMoves(board, 1)
        print('rp: possible moves: ', end="")
        for i in range(len(valid)):
            if valid[i]:
                print(i, end=' ')
        print('')

        while valid[select] != 1:
            select = np.random.randint(self.game.getActionSize())
        print('rp: select ' + str(select))

        b = Board(6)
        b.pieces = np.copy(board)
        b.check_board(select, prefix = "rp: ")
        return select

class OracleAwariPlayer():
    def __init__(self, game, mistake_fraction = 0.0, mistake_max = 4):
        self.game = game
        self.mistake_fraction = mistake_fraction
        self.mistake_max = mistake_max

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        print('op: possible moves: ', end="")
        for i in range(len(valid)):
            if valid[i]:
                print(i, end=' ')
        print('')

        b = Board(6)
        b.pieces = np.copy(board)
        scores = b.oracle_eval_board(prefix = "op: ")
        best = -127
        best_i = 6
        next_best = best
        next_best_i = best_i
        for i in range(6):
            if scores[i] != 127:
                # pick best move, or choose between best ones;
                # allow for mistakes if so desired
                if scores[i] > best or (scores[i] == best and random.random() < 0.5):
                    next_best = best
                    next_best_i = best_i
                    best = scores[i]
                    best_i = i
        select = best_i
        if random.random() < self.mistake_fraction:
            # take other choice, if not absurd:
            if next_best != -127 and (best - next_best) <= self.mistake_max:
                select = next_best_i
                print('op: select suboptimal move')
        print('op: select ' + str(select))

        return select


class HumanAwariPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)

        print('hp: possible moves: ', end="")
        for i in range(len(valid)):
            if valid[i]:
                print(i, end=' ')
        print('')

        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x = 0
            for y in a.split(' '):
                x = int(y)
            a = x if x != -1 else 2 * self.game.n
            if a >= 0 and a <= self.game.n and valid[a]:
                break
            else:
                print('hp: Invalid')

        select = a
        b = Board(6)
        b.pieces = np.copy(board)
        b.check_board(select, prefix = "hp: ")

        return a

class GreedyAwariPlayer():
    def __init__(self, game):
        '''
        :param game:this includes valid move rules,etc.
        '''
        self.game = game

    def play(self, board):
        '''
        :param board: the current configuration of the board
        :return: if more actions have the same value, which is the best one, it returns randomly one action from these
        '''
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        list = []
        max = candidates[0][0]
        for i in range(len(candidates)):
            if candidates[i][0] == max:
                list.append(candidates[i][1])
        select = random.choice(list)

        b = Board(6)
        b.pieces = np.copy(board)
        b.check_board(select, prefix = "gp: ")

        return select


class MinMaxAwariPlayer():
    def __init__(self,game,depth):
       '''
        :param game: the game with the rules
        :param depth: the depth to which alpha beta to search
       '''
       self.game=game
       self.depth=depth

    def play(self,board):
        '''
        :param board: the configuration of the board
        :return: the action from the tuple (action, score) where this action is the best action detected by alfa-beta
        '''

        score = self.minimax((board,-1),self.depth,1,-infinity,+infinity)
        print("mp: minmax at depth " + str(self.depth))
        select = score[0]
        print("mp: select " + str(select))

        b = Board(6)
        b.pieces = np.copy(board)
        b.check_board(select, prefix = "mp: ")

        return select

    def minimax(self,state,depth,player,alfa,beta):
        '''
        :param state: the configuration of the board at current time
        :param depth: depth of the search of alfa-beta
        :param player: which player is currently moving(1-for current player,-1 for adversary)
        :param alfa: the initialization of alfa(here is -infinity)
        :param beta: the initialization of beta(here is +infinity)
        :return: the [action,score] of the best move
        '''

        best = [None, None]

        if player==1:
            best[1]=-infinity
        else:
            best[1]=+infinity

        if self.game.getGameEnded(state[0], player) != 0:
            score = self.game.getGameEnded(state[0], player)
            return [None, score]
        elif depth == 0:
            score = self.game.getScore(state[0], player)
            return [None, score]

        '''
        if depth==0 or self.game.getGameEnded(state[0],player)!=0:
            score=self.game.getGameEnded(state[0],player)
            return [None,score]
        '''

        valids = self.game.getValidMoves(state[0], player)
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard= self.game.getNextState(state[0], player, a)
            score = self.minimax(nextBoard, depth-1, -player,alfa,beta)

            if player==1:
                if score[1] > best[1]:
                    best[1]=score[1]
                    best[0]=a
                alfa=max(alfa,best[1])
                if beta<=alfa:
                    break
            else:
                if score[1]<best[1]:
                    best[1]=score[1]
                    best[0]=a
                beta=min(beta,best[1])
                if beta<=alfa:
                    break
        return best
