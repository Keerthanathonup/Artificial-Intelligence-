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
#                                Milestone 3 with aplha beta pruning and heuristics
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

#Gets number of empty spots in the board
def getNumberOfEmptySpots(board):
    count = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                count += 1
    return count

#Function to get monotonocity Score
#Reference to calculate this score: https://theresamigler.files.wordpress.com/2020/03/2048.pdf
def getMonotonocityScore(boardInput):
    board = copy.deepcopy(boardInput)
    best = -1
    for i in range(0,4):
        current = 0
        for row in range(0,4):
            for col in range(0,3):
                if board[row][col] >= board[row][col+1]:
                    current += 1
        
        for col in range(0,4):
            for row in range(0,3):
                if board[row][col] >= board[row+1][col]:
                    current += 1
        
        if current > best:
            best = current

        #Rotate by 90 degrees
        board = reverse(transpose(board))
    return best
    
#Choose the direction by filtering in the order of
# 1. More empty spots
# 2. Is Monotonous
# 3. Random
# Input: dirToBoard hash map where Direction enum is they key and the board we obtain when swiping in that 
# direction is the value
# Output: direction
def getBestDirection(dirToBoard):
    highestEmptySpotObtainingDirections = []
    maxNumberOfEmptySpots = -1
    if len(dirToBoard.keys()) == 1:
        return list(dirToBoard.keys())[0]
    for direction,board in dirToBoard.items():
        numberOfEmptySpots = getNumberOfEmptySpots(board)
        if numberOfEmptySpots > maxNumberOfEmptySpots:
            highestEmptySpotObtainingDirections = []
            highestEmptySpotObtainingDirections.append(direction)
        elif numberOfEmptySpots == maxNumberOfEmptySpots:
            highestEmptySpotObtainingDirections.append(direction)
    #If there is only one child with highest number of empty spots, return it
    if len(highestEmptySpotObtainingDirections) == 1:
        return highestEmptySpotObtainingDirections[0]
    highestMonotonocityObtainingDirections = []
    highestMonotonousScore = -1
    for direction in highestEmptySpotObtainingDirections:
        score = getMonotonocityScore(dirToBoard[direction])
        if score > highestMonotonousScore:
            highestEmptySpotObtainingDirections = []
            highestMonotonocityObtainingDirections.append(direction)
        elif score == highestMonotonousScore:
            highestMonotonocityObtainingDirections.append(direction)
    if len(highestMonotonocityObtainingDirections) == 1:
        return highestMonotonocityObtainingDirections[0]
    #Choose one randomly if there are more than one direction producing same monotonous score
    return highestMonotonocityObtainingDirections[random.randint(0,len(highestMonotonocityObtainingDirections)-1)]

#Minimax algorithm that recursively identifies the best swipe direction to take
# using alpha beta pruning and heuristics
def minimaxWithAlphaBetaPruning(board, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or board_is_full(board):
        return 0, None

    # Tries to get maximum score of left, right, up & down moves
    if maximizingPlayer:
        maxEval = float('-inf')
        maxEvalDirectionBoardMap = {}
        for (childState, score, direction) in getSwipedStatesAndScores(board):
            if not stateChangedAfterSwipe(board, childState):
                continue
            (eval, _) = minimaxWithAlphaBetaPruning(childState, depth - 1, alpha, beta, False)
            alpha = max(alpha, eval)
            # Get directions with equal scores to maxEvalDirections
            if eval + score > maxEval:
                maxEval = eval + score
                maxEvalDir = direction
                maxEvalDirectionBoardMap = {}
            if eval + score == maxEval:
                maxEvalDirectionBoardMap[direction] = childState
            if beta <= alpha:
                break
        #Get the next best direction to swipe in
        maxEvalDir = getBestDirection(maxEvalDirectionBoardMap)
        return maxEval, maxEvalDir
    #Just puts a 2/4 in such a place that it minmizes the high score
    else:
        minEval = float('inf')
        for childState in getNextStates(board):
            (eval, _) = minimaxWithAlphaBetaPruning(childState, depth - 1, alpha, beta, True)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, None

def milestone_3():
    start = time.time()

    board = getInitialState()
    depth = 7
    totalScore = 0

    #Continue the game until board is not full and board does not contain 2048 
    while not board_is_full(board) and not boardContains2048(board):
        #Find the next best swipe direction using minimax algorithm
        (_, nextBestMoveDir) = minimaxWithAlphaBetaPruning(board, depth, float('-inf'), float('inf'), True)
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