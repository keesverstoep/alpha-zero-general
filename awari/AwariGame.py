from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .AwariLogic import Board
import numpy as np

"""
Game class implementation for the game of Awari.
Based on the TicTacToeGame Evgeny Tyurin which was
based on the OthelloGame by Surag Nair.
Author: Kees Verstoep, Vrije Universiteit Amsterdam
"""

global game_verbose
game_verbose = 0
# game_verbose = 1

class AwariGame(Game):
    def __init__(self, n=6):
        self.n = n
        # self.nplanes = 9
        self.nplanes = 1

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        # y-dimension required for NNet integration

        # For NNet integration we transform the board into an image stack
        # which highlights some useful structural information which would
        # be hard to be derived independently.  AlphagoZero does this too.
        # return (6, 6, 9)
        return (6, 6, self.nplanes)

    def getImageStackSize(self):
        """ Returns size of image stack that is used as input to NNet
        """
        # return 9
        return self.nplanes

    def getImageStack(self, board):
        """ Returns input stack for the given board
        """
        # create image stack that will be an input to NNet 
        n = self.n

        # 2D version for better compatibility, also circular sowing
        # NOTE: channels last for compatibility with other games
        # ORIG:
        # nplanes = 9
        nplanes = self.nplanes
        # main_planes = np.zeros(shape=(6, 6, 9))
        main_planes = np.zeros(shape=(6, 6, nplanes))
        # main images
        #
        # 3 | 10  9  8  7
        # 2 | 11        6
        # 1 |  0        5
        # 0 |  1  2  3  4
        #   +------------
        #      0  1  2  3
        # ind_x = [ 0, 0, 1, 2, 3, 3, 3, 3, 2, 1, 0, 0 ]
        # ind_y = [ 1, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 2 ]
        ind_x = [ 1, 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 1 ]
        ind_y = [ 2, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 3 ]

        # since this implementation uses canonical boards, always
        # assume player is on this side of the board (starting, white)
        player_white = 1
        for pit in range(2 * Board.pits_n):
            i = ind_x[pit]
            j = ind_y[pit]
            if j <= 2:
                if (j == 1 and i >= 1 and i <= 4) or (j == 2 and (i == 1 or i == 4)):
                    if nplanes > 1:
                        main_planes[i][j][1] = board[0][pit]
                        main_planes[i][j][3] = 1
                        main_planes[i][j][5] = (board[0][pit] < 3)
                    if player_white or nplanes == 1:
                        main_planes[i][j][0] = 1
            else:
                if (j == 4 and i >= 1 and i <= 4) or (j == 3 and (i == 1 or i == 4)):
                    if nplanes > 1:
                        main_planes[i][j][2] = board[0][pit]
                        main_planes[i][j][4] = 1
                        main_planes[i][j][6] = (board[0][pit] < 3)
                    if not player_white or nplanes == 1:
                        main_planes[i][j][0] = 1

        # add own pits
        if nplanes > 1:
            main_planes[5][1][7] = board[0][Board.pit_captured_self]
            main_planes[0][2][8] = board[0][Board.pit_captured_other]
        else:
            main_planes[5][1][0] = board[0][Board.pit_captured_self]
            main_planes[0][2][0] = board[0][Board.pit_captured_other]

        #print('board:')
        #print(board)
        #print('main_planes:')
        #print(main_planes)
        return main_planes

    def getActionSize(self):
        # return number of actions: select one of own pits or (forced) pass
        return self.n + 1

    def getNextState(self, board, player, action):
        # Note: board is a regular board, not a canonical one!
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        #
        if game_verbose: print("execute action:" + str(action) + " for " + str(player))
        if action == 2 * self.n:
            # pass; TODO: swap board??
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        b.execute_move(action, player)
        if game_verbose:
            print("new board:")
            display(b.pieces)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        if game_verbose: print('get valid moves of player ' + str(player))
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x in legalMoves:
            valids[x] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if this player won, -1 if player lost
        # NOTE: player may also be -1, then we return result from the
        # perspective of this player
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.pieces[0][Board.pit_no_captures] > 80:
            # HACK: don't get stuck in no-capture loops
            # may need board hashing trick instead
            if b.is_majority(player):
                return 1
            elif b.is_majority(-player):
                return -1
            else:
                # draw
                return 1e-4
        if b.has_legal_moves():
            # TODO: captured stone balance is essential info for board
            # evaluation
            return 0
        # draw has a very little value 
        return 1e-4

    def getCanonicalForm(self, board, player):
        # NOTE: board already is a numpy array here, not a Board!
        # return state if player==1, else return -state if player==-1
        if player == 1:
            # return board.pieces
            #return board
            ret_pieces = np.copy(board)
            return ret_pieces
        else:
            b = Board(self.n)
            b.pieces = np.copy(board)
            mirror = b.mirror()
            ret_pieces = np.copy(mirror.pieces)
            return ret_pieces

    def getSymmetries(self, board, pi):
        # no mirror, or rotational, just the orignal
        l = [(board, pi)]
        return l

    def stringRepresentation(self, board):
        # numpy array (canonical board)
        return board.tostring()

    # for evaluation, just the number of stones captures minus the
    # ones score by the opponent:
    def getScore(self, board, player):
        pits = self.n
        val = board[0][Board.pit_captured_self] - board[0][Board.pit_captured_other]
        if player == 1:
            return val
        else:
            return -val

def display(board):
    pits = Board.pits_n
    print("   ", end="")
    for i in range(pits):
        print (str(board[0][2 * pits - 1 - i]) + " ", end="")
    print("")
    print (str(board[0][Board.pit_captured_other]) + "               " + str(board[0][Board.pit_captured_self]))
    print("   ", end="")
    for i in range(pits):
        print (str(board[0][i]) + " ", end="")
    print("")

