from collections import deque
import copy
import random
from enum import Enum
import stat
import time

# Left Swipe
# input : 4 x 4 matrix before swipe
# output : 4 x 4 matrix after swipe, Score
def leftSwipe(matrix):
	leftSwipedMatrix = []
	score = 0
	i = 0
	while i < len(matrix):
		swipedRow = []
		front = 0
		
		while front < len(matrix[0]) and matrix[i][front] == 0:
			front += 1
		
		if front == 0:
			rear = 0
			front  = 1
		else:
			rear = front - 1

		while front < len(matrix[0]):

			# Move through 0's
			while front < len(matrix[0]):
				if matrix[i][front] == 0:
					front += 1
				else:
					break

			if front < len(matrix[0]):
				if matrix[i][rear] == matrix[i][front]:
					element = matrix[i][rear] * 2
					score += element
					swipedRow.append(element)
					rear = front + 1
					front = front + 2
				else:
					if matrix[i][rear] != 0:
						swipedRow.append(matrix[i][rear])
					rear = front
					front += 1

		if rear < len(matrix[0]):
			swipedRow.append(matrix[i][rear])

		if len(swipedRow) < len(matrix[0]):
			for k in range(len(matrix[0]) - len(swipedRow)):
				swipedRow.append(0)
		
		leftSwipedMatrix.append(swipedRow)

		i += 1

	return leftSwipedMatrix, score


# Right Swipe
# input : 4 x 4 matrix before swipe
# output : 4 x 4 matrix after swipe, Score
def rightSwipe(matrix):
	reversed_matrix = reverse(matrix)
	rightSwapedReversedMatrix, score = leftSwipe(reversed_matrix)
	rightSwipedMatrix = reverse(rightSwapedReversedMatrix)
	return rightSwipedMatrix, score

# Up Swipe
# input : 4 x 4 matrix before swipe
# output : 4 x 4 matrix after swipe, Score
def upSwipe(matrix):
	result_matrix, score = leftSwipe(transpose(reverse(matrix)))
	return reverse(transpose(result_matrix)), score


# Down swipe
# input : 4 x 4 matrix before swipe
# output : 4 x 4 matrix after swipe, Score
def downSwipe(matrix):
	result_matrix, score = leftSwipe(reverse(transpose(matrix)))
	return transpose(reverse(result_matrix)), score	


# Helper functions: reflect
def transpose(matrix_arg):
	matrix_to_transpose = copy.deepcopy(matrix_arg)
	n = len(matrix_to_transpose)
	for i in range(n):
		for j in range(i + 1, n):
			matrix_to_transpose[i][j], matrix_to_transpose[j][i] = matrix_to_transpose[j][i], matrix_to_transpose[i][j]
	return matrix_to_transpose

def reverse(matrix_arg):
	matrix_to_reverse = copy.deepcopy(matrix_arg)
	n = len(matrix_to_reverse)
	for i in range(n):
		for j in range(n // 2):
			matrix_to_reverse[i][j], matrix_to_reverse[i][-j-1] = matrix_to_reverse[i][-j-1], matrix_to_reverse[i][j]
	return matrix_to_reverse


def addTwo(matrix):
	valueSet = 0
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if matrix[i][j] == 0:
				matrix[i][j] = 2
				valueSet = 1
				break
		if valueSet == 1:
			break
	return matrix


################################################################################################################################
#
#                                  Milestone 2
#
################################################################################################################################

class Direction(Enum):
	LEFT = 'L'
	RIGHT = 'R'
	UP = 'U'
	DOWN = 'D'


# Given a current state, gets all possible next states of the board by adding 2/4 at each empty spot
def getNextStates(currentState):
	nextStates = []
	for i in range(len(currentState)):
		for j in range(len(currentState[0])):
			if currentState[i][j] == 0:
				temp = copy.deepcopy(currentState)
				temp[i][j] = 2
				nextStates.append(copy.deepcopy(temp))
				temp[i][j] = 4
				nextStates.append(copy.deepcopy(temp))
	return nextStates

# Get all states where we have two 2's and respective score [(state, score),(state, score)]
def getInitialState():
	temp = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
	rowForFirstTwo = random.randint(0,3)
	colForFirstTwo = random.randint(0,3)
	rowForSecondTwo = random.randint(0,3)
	colForSecondTwo = random.randint(0,3)

	while rowForSecondTwo == rowForFirstTwo and colForSecondTwo == colForFirstTwo:
		rowForSecondTwo = random.randint(0,3)
		colForSecondTwo = random.randint(0,3)

	temp[rowForFirstTwo][colForFirstTwo] = 2
	temp[rowForSecondTwo][colForSecondTwo] = 2
	return temp	
	

def getSwipedStatesAndScores(state):
	stateAndScores = []

	# Get state after Left swipe and score
	leftSwipedState, leftSwipedScore = leftSwipe(state)

	# Get state after right swipe and score
	rightSwipedState, rightSwipedScore = rightSwipe(state)

	# Get state after up swipe and score
	upSwipedState, upSwipedScore = upSwipe(state)

	# Get state after down swipe and score
	downSwipedState, downSwipedScore = downSwipe(state)

	stateAndScores.append((leftSwipedState, leftSwipedScore, Direction.LEFT))
	stateAndScores.append((rightSwipedState, rightSwipedScore, Direction.RIGHT))
	stateAndScores.append((upSwipedState, upSwipedScore, Direction.UP))
	stateAndScores.append((downSwipedState, downSwipedScore, Direction.DOWN))

	#return all states and score
	return stateAndScores

def getMaxScoreInNextStep(swipedStatesAndScores):
	maxScore = 0
	for i in range(len(swipedStatesAndScores)):
		maxScore = max(maxScore, swipedStatesAndScores[i][1])
	return maxScore

def getNextSwipedStateAndScore(currStateAndScore, isRandom):
	validNextStates = []
	currSwipedState, currScore, currDirection = currStateAndScore
	nextStates = getNextStates(currSwipedState)

	if len(nextStates) == 0:
		return (-1,-1,-1)

	for s in nextStates:
		nextSwipedStatesAndScores = getSwipedStatesAndScores(s)
		maxNextStateScore = getMaxScoreInNextStep(nextSwipedStatesAndScores)
		for k in range(len(nextSwipedStatesAndScores)):
			nextSwipedState, nextScore, nextDirection = nextSwipedStatesAndScores[k]
			if (not(isRandom) and (nextScore == maxNextStateScore)) or (isRandom and ((nextScore == maxNextStateScore) and (currScore + maxNextStateScore) > 0)):
				validNextStates.append(((nextSwipedState, nextScore, nextDirection)))

	if len(validNextStates) > 1:
		randCombinationIndex = random.randint(0,len(validNextStates) - 1)
		return validNextStates[randCombinationIndex]
	elif len(validNextStates) == 1:
		return validNextStates[0]
	else:
		return (-1,-1,-1)
	
def boardContains2048(state):
	for i in range(len(state)):
		for j in range(len(state[0])):
			if state[i][j] == 2048:
				return True
	return False


def printGameBoard(myboard, tile_width=4):
    msg = ""
    for i, row in enumerate(myboard):
        # Print row
        element = row[0]  
        element_str = '{:^{width}}'.format(element, width=tile_width)
        msg = msg + element_str

        for element in row[1:]:
            element_str = '|{:^{width}}'.format(element, width=tile_width)  
            msg = msg + element_str

        msg = msg + '\n'

        # Print row divider if its not the last row
        if i is not len(myboard) - 1:
            element_str = '{:-^{width}}'.format("", width=((tile_width + 1) * len(row) - 1))  
            msg = msg + element_str
            msg = msg + '\n'
    print(msg)

#############################################################################################################
#
#                                Milestone 3
#
#############################################################################################################

#Checks if the board is full so that we can't proceed to next swipe
def board_is_full(board):
    for i in range(0, len(board)):
        for j in range(0, len(board[0])):
            if board[i][j] == 0:
                return False
            if i != 0 and board[i][j] == board[i-1][j]:
                return False
            if j != 0 and board[i][j] == board[i][j-1]:
                return False
    return True

#Randomly insert a 2/4 at an empty spot and return the board
def getBoardWithRandomlyInsertedTwoOrFour(board):
    emptySpots = []
    values = [2,4]
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                emptySpots.append((i,j))
    randEmptySpot = emptySpots[random.randint(0,len(emptySpots) - 1)]
    randValue = values[random.randint(0,len(values)-1)]
    newBoard = copy.deepcopy(board)
    newBoard[randEmptySpot[0]][randEmptySpot[1]] = randValue
    return newBoard

#Check if the state of the board changed after swipe or remained same       
def stateChangedAfterSwipe(parent, child):
    for i in range(len(parent)):
        for j in range(len(parent[0])):
            if parent[i][j] != child[i][j]:
                return True
    return False

#Minimax algorithm that recursively identifies the best swipe direction to take
def minimax(board, depth, maximizingPlayer):
    if depth == 0 or board_is_full(board):
        return 0, None

    # Tries to get maximum score of left, right, up & down moves
    if maximizingPlayer:
        maxEval = float('-inf')
        maxEvalDirections = []
        for (childState, score, direction) in getSwipedStatesAndScores(board):
            if not stateChangedAfterSwipe(board, childState):
                continue
            (eval, _) = minimax(childState, depth - 1, False)
            if eval + score > maxEval:
                maxEval = eval + score
                maxEvalDir = direction
                maxEvalDirections = []
            if eval + score == maxEval:
                maxEvalDirections.append(maxEvalDir)
        # If swiping in multiple directions results in same score, choose one of them randomly
        if len(maxEvalDirections) != 0:
            maxEvalDir = maxEvalDirections[random.randint(0,len(maxEvalDirections)-1)]
        return maxEval, maxEvalDir
    #Just puts a 2/4 in such a place that it minmizes the high score
    else:
        minEval = float('inf')
        for childState in getNextStates(board):
            (eval, _) = minimax(childState, depth - 1, True)
            minEval = min(minEval, eval)
        return minEval, None

def milestone_3():
    start = time.time()

    # Get Initial State of the board
    board = getInitialState()
    depth = 3
    totalScore = 0

    #Continue the game until board is not full and board does not contain 2048 
    while not board_is_full(board) and not boardContains2048(board):
        #Find the next best swipe direction using minimax algorithm
        (_, nextBestMoveDir) = minimax(board, depth, True)
        if nextBestMoveDir == Direction.LEFT:
            (board, score) = leftSwipe(board)
        elif nextBestMoveDir == Direction.RIGHT:
            (board, score) = rightSwipe(board)
        elif nextBestMoveDir == Direction.DOWN:
            (board, score) = downSwipe(board)
        elif nextBestMoveDir == Direction.UP:
            (board, score) = upSwipe(board)
        # insert 2 or 4 randomly at a location
        board = getBoardWithRandomlyInsertedTwoOrFour(board)
        totalScore += score	

    #Print the game output
    equal_to_pattern = "=" * 17
    print("\n" + equal_to_pattern + "=  FINAL BOARD  =" + equal_to_pattern + "\n")
    printGameBoard(board)
    print("Depth d :", depth)
    print("Maximum score :", totalScore)
    end = time.time()
    print("Total run time of the game :", end - start, "\n")
    print(equal_to_pattern * 3 + "\n")

milestone_3()