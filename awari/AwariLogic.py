# from awari.AwariGame import display
import numpy as np
# to ask the oracle:
from subprocess import check_output


'''
Board class for the game of Awari.
Default board size is 2x6 plus two end pits, one for the players
Board data:
  count of stones in each pit

Based on the board for the game of Othello by Eric P. Nichols.
'''

next = [
    [ 1, 2, 3, 4, 5, 7, 7, 8,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 8, 8,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 9,  9, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 11, 11, 0, ],
    [ 1, 2, 3, 4, 5, 6, 7, 8,  9, 10,  0, 0, ],
]

global board_verbose
board_verbose = 0

# from bkcharts.attributes import color
class Board():
    pits_n = 6
    pits_total = 2 * pits_n
    pits_alloc = pits_total + 4
    pit_captured_self = pits_total + 2
    pit_captured_other = pits_total + 3
    pit_no_captures = pits_total + 1

    def __init__(self, n=6):
        "Set up initial board configuration."

        self.n = n
        # shortcut, not expected to change in practice
        assert self.n == Board.pits_n

        # Create the empty board array.
        #self.pieces = [None] * Board.pits_alloc
        #for i in range(Board.pits_total):
        #    self.pieces[i] = [0]
        #    self.pieces[i][0] = 4
        #for i in range(Board.pits_total, Board.pits_alloc):
        #    self.pieces[i] = [0]
        #    self.pieces[i][0] = 0
        self.pieces = [[4, 4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4, 4,
                        0, 0,
                        0, 0]]

    # add [] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def can_move(self, color):
        if color == 1:
            for i in range(Board.pits_n):
                if self.pieces[0][i] > 0:
                    return True
        else:
            for i in range(Board.pits_n, Board.pits_total):
                if self.pieces[0][i] > 0:
                    return True
        return False

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.

        if board_verbose:
            print('get legal moves of color ' + str(color) + ' for ')
            self.display()

        # NOTE: board is in canonical form

        # first find moves that allow the opponent to respond
        for i in range(Board.pits_n):
            if self.pieces[0][i] != 0:
                # check that when doing this move, the opponent can respond
                child = self.sow(i, color)
                board = child.mirror()
                if board.can_move(-color):
                    if board_verbose: print('add move ' + str(i))
                    moves.add(i)
                else:
                    if board_verbose: print('skip move ' + str(i))

        if len(moves) == 0:
            # retry, now adding moves that will allow no response
            for i in range(Board.pits_n):
                if self.pieces[0][i] != 0:
                    if board_verbose: print('add move ' + str(i))
                    moves.add(i)

        return list(moves)

    def has_legal_moves(self):
        # passing is also legal, so just check there are uncaptured stones
        return (self.pieces[0][Board.pit_captured_self] + self.pieces[0][Board.pit_captured_other]) < (4 * Board.pits_total)
    
    def is_win(self, color):
        """Check whether the given player has majority of the stones
        @param color (1=white,-1=black)
        """
        # Playing to the end may cause very deep recursions, but determining
        # a win does not require this, more than half the stones is enough
        if color == 1:
            win = (self.pieces[0][Board.pit_captured_self] > 4 * Board.pits_n)
        else:
            win = (self.pieces[0][Board.pit_captured_other] > 4 * Board.pits_n)
        return win

    def is_majority(self, color):
        """Check whether the given player has majority of the stones, supposing remaining
        stones get divided due to repetition of board configuration.
        @param color (1=white,-1=black)
        """
        # Playing to the end may cause very deep recursions, currently take shortcut
        if color == 1:
            return (self.pieces[0][Board.pit_captured_self] > self.pieces[0][Board.pit_captured_other])
        else:
            return (self.pieces[0][Board.pit_captured_other] > self.pieces[0][Board.pit_captured_self])

    def mirror(self):
        board = Board(self.n)

        for i in range(Board.pits_n):
             board.pieces[0][i] = self.pieces[0][i + Board.pits_n]
             board.pieces[0][i + Board.pits_n] = self.pieces[0][i]
        board.pieces[0][Board.pit_no_captures] = self.pieces[0][Board.pit_no_captures]
        # TODO: should the self/other fields also not be swapped???
        #board.pieces[0][Board.pit_captured_self] = self.pieces[0][Board.pit_captured_self]
        #board.pieces[0][Board.pit_captured_other] = self.pieces[0][Board.pit_captured_other]
        board.pieces[0][Board.pit_captured_self] = self.pieces[0][Board.pit_captured_other]
        board.pieces[0][Board.pit_captured_other] = self.pieces[0][Board.pit_captured_self]

        # DEBUG
        # print('mirror pieces')
        # print(type(board.pieces))
        # print(board)
        # END DEBUG

        return board
            
    def sow(self, move, color):
        # assumes a canonical board as input and returns a canonical board
        pit1 = move
        seeds = self.pieces[0][pit1]
        pit2 = pit1 + 6
        capture = 0
        oldseeds = seeds
        next_ptr = next[pit1]

        child = self.mirror()
        # take away seeds
        child.pieces[0][pit2] = 0

        # sow them
        while seeds > 0:
            pit2 = next_ptr[pit2]
            child.pieces[0][pit2] += 1
            seeds -= 1

        captures = 0
        while (pit2 < 6 and pit2 >= 0) and ((child.pieces[0][pit2] == 2) or (child.pieces[0][pit2] == 3)):
            # capture
            captures += child.pieces[0][pit2]
            child.pieces[0][pit2] = 0
            pit2 -= 1

        if captures > 0:
            if board_verbose: print('captured ' + str(captures) + ' by ' + str(color))
            #if color == 1:
            #    child.pieces[0][Board.pit_captured_self] += captures
            #else:
            #    child.pieces[0][Board.pit_captured_other] += captures
            # board is mirrored, so pit is in the other of the child:
            child.pieces[0][Board.pit_captured_other] += captures
            # monitor no-progress cycles
            child.pieces[0][Board.pit_no_captures] = 0
        else:
            child.pieces[0][Board.pit_no_captures] += 1

        return child

    def display(self):
        print(self)
        print(self.pieces)
        for i in range(Board.pits_n):
            print (str(self.pieces[0][i]) + " ", end="")
        print(" ", end="")
        for i in range(Board.pits_n, Board.pits_total):
            print (str(self.pieces[0][i]) + " ", end="")
        print(" ", end="")
        print (str(self.pieces[0][Board.pit_captured_self]) + " ", end="")
        print (str(self.pieces[0][Board.pit_captured_other]) + " ", end="")
        print (str(self.pieces[0][Board.pit_no_captures]) + " ", end="")
        print("")

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the player to play (1=white,-1=black)
        """
        # NOTE: board is native/non-canonical, but move is canonical!

        # DEBUG
        #print('execute_move: board ')
        #print(type(self))
        #print(type(self.pieces))
        # END DEBUG

        if board_verbose:
            print('execute_move: board')
            self.display()
            print('move ' + str(move) + ' color ' +str(color))
        if move < Board.pits_n:
            # sow() assumes a canoncial board as input and
            # returns a canonical *child* board!
            if color == 1:
                 board = self
            else:
                 board = self.mirror()
            child = board.sow(move, color)
            if board_verbose:
                print('execute_move: new board:')
                child.display()
            # child is canonical again, but have to return it native order
            if color == 1:
                child = child.mirror()
            self.pieces = np.copy(child.pieces)
            if board_verbose:
                print('execute_move: new board')
                self.display()
        else:
            assert move == Board.pits_n
            # cannot move; all remaining stones on the board for opponent
            stones = 0
            for i in range(Board.pits_total):
                num = self.pieces[0][i]
                # NOTE: BUG FIX:
                if num > 0:
                    self.pieces[0][i] = 0
                    stones += num
            # add to opponent
            if color == 1:
                self.pieces[0][Board.pit_captured_other] += stones
            else:
                self.pieces[0][Board.pit_captured_self] += stones
            if board_verbose:
                print('end board')
                self.display()

        # DEBUG
        #print('execute_move: return board ')
        #print(type(self))
        #print(type(self.pieces))
        # END DEBUG

    def oracle_eval_board(self, prefix):
        # board should already be canonical?
        # prefix is used to distinguish diagnostics for different players

        command = "/home/verstoep/Projects/Awari/awari/db_lookup "
        for i in range(12):
            command += " %d" % self.pieces[0][i]
        print(prefix + command)
        outputs = check_output(command, encoding='UTF-8', universal_newlines=True, shell=True)
        lines = outputs.splitlines()
        for line in lines:
            if line.startswith("children scores"):
                words = line.split()
                scores = [127, 127, 127, 127, 127, 127]
                if len(words) == 8:
                    for i in range(6):
                        scores[i] = int(words[i + 2])
                print(prefix + "scores: " + str(scores))
                return scores
        return []

    def check_board(self, select, prefix):
        self.curPlayer = 1
        moves = self.get_legal_moves(self.curPlayer)
        print(prefix + 'legal moves:' + str(moves))
        print(prefix + 'select: ' + str(select))
        scores = self.oracle_eval_board(prefix)

        best = -127
        best_i = -1
        for i in range(6):
            if scores[i] == 127 and i in moves:
                print(prefix + "*** oracle invalidated move " + i);
            if scores[i] != 127 and not i in moves:
                print(prefix + "*** neural net invalidated move " + i);
            if scores[i] != 127 and scores[i] > best:
                best = scores[i]
                best_i = i
        if select > 5:  # pass
            if best != -127:
                print(prefix + 'unnecessary pass')
        elif scores[select] == best:
            print(prefix + 'best move selected')
        else:
            print(prefix + 'suboptimal move selected, diff ' + str(best - scores[select]))
