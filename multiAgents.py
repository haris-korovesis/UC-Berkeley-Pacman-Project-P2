# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #Get the ghost pos
        ghostPos = 0
        #Dont go closer than 3
        minDistance = 3
        for ghost in newGhostStates:
            ghostPos = util.manhattanDistance(newPos,ghost.getPosition())
            # Check if the closest ghost in more than 3 away
            temp = min(minDistance,ghostPos)
            # If its not the score goes to 0 -> keep going
            # If its closer than 3 the score goes negative -> avoid the ghost
            ghostPos = (temp - minDistance) * 100
            #adjust the new score
            ghostPos += ghostPos




        #Set food pos
        foodPos = 0
        foodList = newFood.asList()
        # The further away is the food , the smaller the point reward 
        for food in foodList:
            foodPos = 1/float(util.manhattanDistance(newPos,food))


        #Adjust the score 
        CurrentScore = successorGameState.getScore()
        score = CurrentScore + foodPos + ghostPos
        return score

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def MinValue(gameState, agentIndex, depth):

            # Get the available actions
            ghostActions = gameState.getLegalActions(agentIndex)

            # If you havent any legal actions return static evaluation of Node
            if not ghostActions:
                return self.evaluationFunction(gameState)

            minimumValue = ["",99999]
            # In a tuple keep the min and the action with it
            for action in ghostActions:
                tempState = gameState.generateSuccessor(agentIndex,action)
                curr = MinMax(tempState,agentIndex+1,depth)
                if type(curr) is not list:
                    temp = curr
                else:
                    temp = curr[1]

                if temp < minimumValue[1]:
                    minimumValue = [action,temp]

            return minimumValue


        def MaxValue(gameState, agentIndex, depth):

            # Get the available actions
            LegalActions = gameState.getLegalActions(agentIndex)

            # If you havent any legal actions return static evaluation of Node
            if not LegalActions:
                return self.evaluationFunction(gameState)

            maximumValue = ["",-99999]
            # keep the max in a tuple and return it along with the available action
            for action in LegalActions:
                tempState = gameState.generateSuccessor(agentIndex,action)
                curr = MinMax(tempState,agentIndex+1,depth)
                if type(curr) is not list:
                    temp = curr
                else:
                    temp = curr[1]

                if temp > maximumValue[1]:
                    maximumValue = [action,temp]

            return maximumValue

        def MinMax(gameState, agentIndex, depth):
            
            if agentIndex >= gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            # Check if we have lost
            if (depth == self.depth or gameState.isWin() or gameState.isLose() ) :
                return self.evaluationFunction(gameState)
                # If pacman
            elif agentIndex == 0:
                return MaxValue(gameState,agentIndex,depth)
                #If ghost
            else:
                return MinValue(gameState,agentIndex,depth)

        
        totalActions = MinMax(gameState,0,0)
        
        return totalActions[0]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def MinValue(gameState, agentIndex, depth, a, b):

            # Get the available actions
            ghostActions = gameState.getLegalActions(agentIndex)

            # If you havent any legal actions return static evaluation of Node
            if not ghostActions:
                return self.evaluationFunction(gameState)

            # give minimum the highest possible value
            # find the minimum value and check if a and b need to be changed
            minimumValue = ["", float("inf")]
            for action in ghostActions:
                tempState = gameState.generateSuccessor(agentIndex,action)
                curr = MinMax(tempState,agentIndex+1,depth, a, b)
                if type(curr) is not list:
                    temp = curr
                else:
                    temp = curr[1]

                if temp < minimumValue[1]:
                    minimumValue = [action,temp]

                if temp < a:
                    return [action,temp]

                b = min(b, temp)

            return minimumValue


        def MaxValue(gameState, agentIndex, depth, a, b):

            # Get the available actions
            LegalActions = gameState.getLegalActions(agentIndex)

            # If you havent any legal actions return static evaluation of Node
            if not LegalActions:
                return self.evaluationFunction(gameState)

            maximumValue = ["", -float("inf")]

            for action in LegalActions:
                tempState = gameState.generateSuccessor(agentIndex,action)
                curr = MinMax(tempState,agentIndex+1,depth,a,b)
                if type(curr) is not list:
                    temp = curr
                else:
                    temp = curr[1]

                if temp > maximumValue[1]:
                    maximumValue = [action,temp]

                if temp > b:
                    return [action,temp]

                a = max(a,temp)

            return maximumValue


        def MinMax(gameState, agentIndex, depth, a, b):
            
            if agentIndex >= gameState.getNumAgents():
                depth += 1
                agentIndex = 0
            
            if (depth == self.depth or gameState.isWin() or gameState.isLose() ) :
                return self.evaluationFunction(gameState)
            elif agentIndex == 0:
                return MaxValue(gameState,agentIndex,depth,a,b)
            else:
                return MinValue(gameState,agentIndex,depth,a,b)


        totalActions = MinMax(gameState,0,0,-float("inf"),float("inf"))
        return totalActions[0]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"


        def MaxValue(gameState, depth):

            # Get Pacmans available actions
            availableActions = gameState.getLegalActions(0)

            if not availableActions or gameState.isWin() or gameState.isLose() or depth == self.depth :
                return (self.evaluationFunction(gameState),None)

            score = -(float("inf"))
            move = None
            # find the maxscore and return it along with the move 
            for action in availableActions:
                temp = MinValue(gameState.generateSuccessor(0,action),1,depth)
                temp = temp[0]

                if score < temp:
                    score,move = temp,action
                    maximuScore = [score,move] 

            return maximuScore


        def MinValue(gameState, agentIndex, depth):

            #Get the available actions for each ghost
            availableActions = gameState.getLegalActions(agentIndex)

            if not availableActions:
                return (self.evaluationFunction(gameState),None)

            move = 0
            totalAgents = gameState.getNumAgents()
            for action in availableActions:
                # if pacman
                if agentIndex == totalAgents - 1:
                    newScore = MaxValue(gameState.generateSuccessor(agentIndex,action),depth+1)
                else:
                    newScore = MinValue(gameState.generateSuccessor(agentIndex,action),agentIndex+1,depth)

                newScore = newScore[0]
                avg = newScore/len(availableActions)
                move += avg

            minimumScore = [move,None]
            return minimumScore


        tempValue = MaxValue(gameState,0)[1]
        return tempValue


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    #get basic info
    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood()
    foodList = foodPos.asList()
    ghostPos = currentGameState.getGhostStates()

    #get score
    score = currentGameState.getScore()

    # Lower rewards for food further away
    for food in foodList:
        distance = manhattanDistance(pacmanPos,food)
        score += (1/float(distance))


    #Ghost scores
    for ghost in ghostPos:
        temp = ghost.getPosition()
        distance = manhattanDistance(pacmanPos,temp)

        if distance == 0:
            continue
        
        if distance < 3:
            score += 5*(1/float(distance))
        else:
            score =+ 1/float(distance)


    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
