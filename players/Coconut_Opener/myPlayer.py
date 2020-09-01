# This file will be used in the competition
# Please make sure the following functions are well defined

# Each player is given 5 seconds when a new round started
# If exceeds 5 seconds, all your code will be terminated and 
# you will receive a timeout warning

# Each player is given 1 second to select next best move
# If exceeds 5 seconds, all your code will be terminated, 
# a random action will be selected, and you will receive 
# a timeout warning

from advance_model import *
from utils import *
import copy
import numpy
import time
import time
import random
import math


class myPlayer2(AdvancePlayer):
    
    # initialize
    # The following function should not be changed at all
    def __init__(self, _id):
        super().__init__(_id)
        
    # Function used for logging information of the game for debugging
    def record(self, string):
        log = open('log', 'a')
        log.write(str(string))
        log.write('\n')
        log.close()
    
    # Get potential score based on the state of grid scheme of the player.
    def adjacent(self, grid, i, j, grid_scheme):
        totalTile = 0
        for x in range(5):
            for y in range(5):
                totalTile = totalTile + grid[x][y]
        self.record(totalTile)

        if grid[i][j] == 1:
            return 0

        result = 0
        adjx = 0
        adjy = 0
        sameColor = 0

        for x in range(0, i):
            if grid[i - 1 - x][j] == 1:
                adjx = adjx + 1
            else:
                break
        for x in range(i + 1, 5):
            if grid[x][j] == 1:
                adjx = adjx + 1
            else:
                break

        for y in range(0, j):
            if grid[i][j - 1 - y] == 1:
                adjy = adjy + 1
            else:
                break
        for y in range(j + 1, 5):
            if grid[i][y] == 1:
                adjy = adjy + 1
            else:
                break

        color = None
        for y in Tile:
            if grid_scheme[i][y] == j:
                color = y

        for x in range(5):
            y = int(grid_scheme[x][color])
            sameColor = sameColor + grid[x][y]

        if sameColor == 4:
            result = result + 10
        if sameColor == 3:
            result = result + 1

        if adjx == 4:
            result = result + 7
        if adjx == 3:
            result = result + 0.4
        if adjy == 4:
            result = result + 2

        if adjx > 0:
            result = result + 1
        if adjy > 0:
            result = result + 1

        result = result + adjx + adjy
        if result == 0:
            result = 1

        if i < 4 and i > 0 and j < 4 and j > 0 and totalTile < 10:
            result = result + 0.1

        return result
    
    # Set score bonus for each position based on the grid scheme.
    def StartRound(self, game_state):
        self.moveNum = 0
        self.grid_bonus = numpy.zeros((2, 5, 5))

        grid = [[[]], [[]]]
        grid[self.id] = game_state.players[self.id].grid_state
        grid[1 - self.id] = game_state.players[1 - self.id].grid_state

        for p in range(2):
            for i in range(5):
                for j in range(5):
                    self.grid_bonus[p][i][j] = self.adjacent(grid[p], i, j, game_state.players[self.id].grid_scheme)

        return None

    # Fuction that calculate the poteintail reward based on the player id, game state and move executed.
    def score(self, move, game_state, playerID):
        result = 0
        mid, fid, tgrab = move
        player = game_state.players[playerID]

        floors = 0
        for tile in player.floor:
            floors = floors + tile

        row = tgrab.pattern_line_dest
        if row == -1:
            base = 0
        else:
            column = int(player.grid_scheme[row][tgrab.tile_type])
            base = self.grid_bonus[playerID][row][column]

        result = result - tgrab.num_to_floor_line * 400 * (floors + 1)
        result = result + tgrab.num_to_pattern_line * 149 * base

        emptyLine = 0
        for n in player.lines_number:
            if n == 0:
                emptyLine = emptyLine + 1

        if row > -1 and row + 1 == player.lines_number[row] + tgrab.num_to_pattern_line:
            result = result + 200 * base
            if row == 0:
                result = result - 50 * base
                if self.moveNum < 3:
                    result = result - 100 * base
        elif player.lines_number[row] == 0:
            result = result - 151 * (5 - emptyLine)

        return result
    
    # Node class
    class treeNode():
        
        # Initialize
        def __init__(self, gamestate, parent, move, pid, moves):
            self.state = gamestate
            self.move = move
            self.moves = moves
            self.usedmoves = []
            self.playerid = pid
            self.isTerminal = not self.state.TilesRemaining()
            self.isFullyExpanded = self.isTerminal
            self.parent = parent
            self.numVisits = 0
            self.totalReward = 0
            self.children = {}
    
    # MCTs simulation step, random select move from 
    def Simulate(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded or len(node.moves) == len(node.usedmoves):
                node = self.getBestChild(node, 1 / math.sqrt(2))
            else:
                movelist = []
                for move in node.moves:
                    if move not in node.usedmoves:
                        movelist.append(move)
                move = random.choice(movelist)
                node.usedmoves.append(move)
                foe, foeMoves, nextState = self.nextstate(move, node.playerid, node.state)
                node = self.treeNode(nextState, node, move, foe, foeMoves)
        return node.state

    # MCTs one search round
    def executeRound(self):
        node = self.selectNode(self.root)
        state = self.Simulate(node)
        roundend = copy.deepcopy(state)
        roundend.ExecuteEndOfRound()
        my_score = roundend.players[self.id].score + roundend.players[self.id].EndOfGameScore()
        foe_score = roundend.players[1-self.id].score + roundend.players[1-self.id].EndOfGameScore()
        reward = my_score - foe_score
        self.backpropogate(node, reward)
    
    # Return a child node which has highest nodeValue for a given node 
    def getBestChild(self, node, value):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = 1 * child.totalReward / child.numVisits + value * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)
    
    # MCTs Selection step, return a node to be simulated
    def selectNode(self, node):
        while not node.isTerminal:
            if len(node.children) == 0:
                return self.expand(node)
            elif random.uniform(0, 1) < .5:
                node = self.getBestChild(node, 1 / math.sqrt(2))
            else:
                if node.isFullyExpanded == False:
                    return self.expand(node)
                else:
                    node = self.getBestChild(node, 1 / math.sqrt(2))
        return node
        
    # Expand the given node and return a child node
    def expand(self, node):
        moves = node.moves
        for action in moves:
            if action not in node.children:
                foe, foeMoves, nextState = self.nextstate(action, node.playerid, node.state)
                newNode = self.treeNode(nextState, node, action, foe, foeMoves)
                newNode.totalReward = self.score(action, node.state, node.playerid)/60
                node.children.update({action: newNode})
                if len(moves) == len(node.children):
                    node.isFullyExpanded = True
                return newNode
    
    # Based on the given move, player id and game state, execute the move and return the next playerid,
    # next game state and available moves for next player.
    def nextstate(self, move, pid, gs):
        nextState = copy.deepcopy(gs)
        nextState.ExecuteMove(pid, move)
        foe = nextState.players[1 - pid]
        foeid = 1-pid
        foeMoves = foe.GetAvailableMoves(nextState)
        return foeid, foeMoves, nextState
    
    # Get the best move from the given node.
    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
    
    # MCTs backpropogation step, backpropogate number of visit, total reward and child node all the way to the root
    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            if node != self.root:
                node.parent.children.update({node.move: node})
            node = node.parent
    
    # MCTs search based on timelimit and given information of the game
    def MctsSearch(self, pid, timelimit, moves, game_state):
        self.root = self.treeNode(game_state, None, None, pid, moves)
        timeLimit = time.time() + timelimit
        while time.time() < timeLimit:
            self.executeRound()
        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def SelectMove(self, moves, game_state):

        timelimit = 0.5
        bset_move = self.MctsSearch(self.id, timelimit, moves, game_state)
        return bset_move

class myPlayer(AdvancePlayer):

    # initialize
    # The following function should not be changed at all
    def __init__(self,_id):
        super().__init__(_id)

    # Get potential score based on the state of grid scheme of the player.
    def adjacent(self, grid, i, j, grid_scheme):
        totalTile = 0
        for x in range(5):
            for y in range(5):
                totalTile = totalTile + grid[x][y]
        self.record(totalTile)
        
        if grid[i][j] == 1:
            return 0

        result = 0
        adjx = 0
        adjy = 0
        sameColor = 0
        
        for x in range(0,i):
            if grid[i-1-x][j]==1:
                adjx = adjx + 1
            else:
                break
        for x in range(i+1,5):
            if grid[x][j]==1:
                adjx = adjx + 1
            else:
                break
        
        for y in range(0,j):
            if grid[i][j-1-y]==1:
                adjy = adjy + 1
            else:
                break
        for y in range(j+1,5):
            if grid[i][y]==1:
                adjy = adjy + 1
            else:
                break

        color = None
        for y in Tile:
            if grid_scheme[i][y] == j:
                color = y
        
        for x in range(5):
            y = int(grid_scheme[x][color])
            sameColor = sameColor + grid[x][y]
            
        if sameColor == 4:
            result = result + 10
        if sameColor == 3:
            result = result + 1

        if adjx == 4:
            result = result + 7
        if adjx == 3:
            result = result + 0.4
        if adjy == 4:
            result = result + 2

        if adjx > 0:
            result = result + 1
        if adjy > 0:
            result = result + 1
        
        result = result + adjx + adjy
        if result == 0:
            result = 1
        
        if i < 4 and i > 0 and j < 4 and j > 0 and totalTile < 10:
            result = result + 0.1

        return result
        
    # Set score bonus for each position based on the grid scheme.
    def StartRound(self,game_state):
        self.moveNum = 0
        self.grid_bonus = numpy.zeros((2,5,5))

        grid = [[[]],[[]]]
        grid[self.id] = game_state.players[self.id].grid_state
        grid[1-self.id] = game_state.players[1-self.id].grid_state
        
        for p in range(2):
            for i in range(5):
                for j in range(5):
                    self.grid_bonus[p][i][j] = self.adjacent(grid[p], i, j, game_state.players[self.id].grid_scheme)

        self.m2 = myPlayer2(self.id)
        self.m2.StartRound(game_state)
        return None

    # Function used for logging information of the game for debugging
    def record(self, string):
        log = open('log', 'a')
        log.write(str(string))
        log.write('\n')
        log.close()
        
    # Fuction that calculate the poteintail reward based on the player id, game state and move executed.
    def score(self, move, game_state, playerID):
        result = 0
        mid, fid, tgrab = move
        player = game_state.players[playerID]

        floors = 0
        for tile in player.floor:
            floors = floors + tile

        row = tgrab.pattern_line_dest
        if row == -1:
            base = 0
        else:
            column = int(player.grid_scheme[row][tgrab.tile_type])
            base = self.grid_bonus[playerID][row][column]

        result = result - tgrab.num_to_floor_line * 400 * (floors + 1)
        result = result + tgrab.num_to_pattern_line * 149 * base

        emptyLine = 0
        for n in player.lines_number:
            if n == 0:
                emptyLine = emptyLine + 1
        
        if row > -1 and row + 1 == player.lines_number[row] + tgrab.num_to_pattern_line:
            result = result + 200 * base
            if row == 0:
                result = result - 50 * base
                if self.moveNum < 3:
                    result = result - 100 * base
        elif player.lines_number[row] == 0:
            result = result - 151 * (5 - emptyLine)
        
        return result
    
    # Based on the given move, player id and game state, execute the move and return
    # the next game state and available moves for next player.
    def expand(self, move, pid, gs):
        nextState = copy.deepcopy(gs)
        nextState.ExecuteMove(pid, move)
        foe = nextState.players[1-pid]
        foeMoves = foe.GetAvailableMoves(nextState)
        return foeMoves, nextState
    
    # Reduce the number of available moves, keep moves with reletively higher potential scores 
    def reduce(self, moves, pid, gs, number):

        if number > len(moves) or number == len(moves):
            return moves

        scores = []
        result = []
        for m in moves:
            scores.append(self.score(m, gs, pid))
        
        for n in range(number):
            maxI = None
            for i in range(len(moves)):
                if (moves[i] not in result) and (maxI == None or scores[maxI] < scores[i]):
                    maxI = i
            result.append(moves[maxI])
        return result
    
    # Return the best move for a given game state using Depth First Search based on the potentail scores.
    def DFS(self, moves, pid, gs, expandNumber, reduceNumber):
        if expandNumber == 0:
            best = None
            bestMove = None
            for m in moves:
                mScore = self.score(m, gs, pid)
                if best == None or mScore > best:
                    best = mScore
                    bestMove = m
            return bestMove, best
        else:
            best = None
            bestMove = None
            for m in moves:
                foeMoves, nextState = self.expand(m, pid, gs)
                foeMoves = self.reduce(foeMoves, 1-pid, nextState, reduceNumber)
                if len(foeMoves) == 0:
                    mScore = self.score(m, gs, pid)
                else:
                    foeMove, foeScore = self.DFS(foeMoves, 1-pid, nextState, expandNumber-1, reduceNumber)
                    mScore = self.score(m, gs, pid) - foeScore * 0.75
                if best == None or mScore > best:
                    best = mScore
                    bestMove = m
            if bestMove == None:
                self.record("None")
            return bestMove, best
    
    # Return the best move for a given game state, if the game will end in current round, use MCTs, else use DFS
    def SelectMove(self, moves, game_state):
        
        if self.willEnd(game_state):
            return self.m2.SelectMove(moves, game_state)

        self.moveNum = self.moveNum + 1
        reduceNum = 100
        if len(moves) > 40:
            expand = 1
        elif len(moves) > 10:
            expand = 2
            reduceNum = 10
        elif len(moves) > 6:
            expand = 3
            reduceNum = 10
        else:
            expand = 5
            reduceNum = 6

        bestMove, bestScore = self.DFS(moves, self.id, game_state, expand, reduceNum)
        return bestMove
    
    # Return boolean whether the game will end in current round
    def willEnd(self, game_state):
        for p in range(2):
            player = game_state.players[p]
            for row in range(5):
                full = 0
                for color in range(5):
                    column = int(player.grid_scheme[row][color])
                    full = full + player.grid_state[row][column]
                if row + 1 == player.lines_number[row]:
                    full = full + 1
                if full == 5:
                    return True
        return False

