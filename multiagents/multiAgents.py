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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        score = successorGameState.getScore()
        # predefined reward if pacman is in a food position
        food_reward = 5

        # calculating the manhattanDistance from current position to every position where we have food
        foodList = newFood.asList()
        food_manhattan = [manhattanDistance(newPos, each) for each in foodList]

        # if we are already in a food pellet we'll add a fixed food_reward to the score
        if newPos in foodList:
            score += food_reward

        # if there is no food left we return score else we calculate the distance to the closest food
        if(len(food_manhattan) != 0):
            food_near = min(food_manhattan)
        else:
            return score

        ghost_positions = successorGameState.getGhostPositions()
        ghost_manhattan = [manhattanDistance(newPos, ghost) for ghost in ghost_positions]
        ghost_near = min(ghost_manhattan)

        # if ghost is closer than 2 steps we try to reduce our score
        # in order to force our agent to avoid the ghost
        if ghost_near < 2:
            score -= 100

        # we return an estimate value, which is the reward to eat food divided by the distance reaching the food
        return score + (food_reward / food_near)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        action, _ = self.minimax(0, 0, gameState)  # Get the action and score for pacman (agent_index=0)
        return action  # Return the action to be done as per minimax algorithm

    def minimax(self, current_depth, agent_index, gameState):
        '''
        Returns the best score for an agent using the minimax algorithm. For max player (agent_index = 0), the best
        score is the maximum score among its successor states and for the min player (agent_index != 0), the best
        score is the minimum score among its successor states. Recursion ends if there are no successor states
        available or current_depth equals the max depth to be searched until.
        :param current_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :return: action, score
        '''
        # Roll over agent index and increase current depth if all agents have finished playing their turn in a move
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            current_depth += 1
        # Return the value of evaluationFunction if max depth is reached
        if current_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        # Initialize best_score and best_action with None
        best_score, best_action = None, None
        if agent_index == 0:  # If it is max player's (pacman) turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of pacman
                # Get the minimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost) turn now
                # Pass the new game state generated by pacman's `action`
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(current_depth, agent_index + 1, next_game_state)
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
        else:  # If it is min player's (ghost) turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of ghost agent
                # Get the minimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost or pacman) turn now
                # Pass the new game state generated by ghost's `action`
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(current_depth, agent_index + 1, next_game_state)
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
        # If it is a leaf state with no successor states, return the value of evaluationFunction
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  # Return the best_action and best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float('inf')
        action, _ = self.alpha_beta(0, 0, gameState, -inf, inf)  # Get the action and score for pacman (max)
        return action  # Return the action to be done as per alpha-beta algorithm
    
    def alpha_beta(self, current_depth, agent_index, gameState, alpha, beta):
        '''
        Returns the best score for an agent using the alpha-beta algorithm. For max player (agent_index = 0), the best
        score is the maximum score among its successor states and for the min player (agent_index != 0), the best
        score is the minimum score among its successor states. Recursion ends if there are no successor states
        available or current_depth equals the max depth to be searched until. If alpha > beta, we can stop generating
        further successors and prune the search tree.
        :param current_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :param alpha: the alpha value of the parent (float)
        :param beta: the beta value of the parent (float)
        :return: action, score
        '''
        # Roll over agent index and increase current depth if all agents have finished playing their turn in a move
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            current_depth += 1
        # Return the value of evaluationFunction if max depth is reached
        if current_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        # Initialize best_score and best_action with None
        best_score, best_action = None, None
        if agent_index == 0:  # If it is max player's (pacman) turn
            legalActions = gameState.getLegalActions(agent_index)
            for action in legalActions:  # For each legal action of pacman
                # Get the minimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost) turn now
                # Pass the new game state generated by pacman's `action` and the current alpha and beta values
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.alpha_beta(current_depth, agent_index + 1, next_game_state, alpha, beta)
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                # Update the value of alpha
                alpha = max(alpha, score)
                # Prune the tree if alpha is greater than beta
                if alpha > beta:
                    break
        else:  # If it is min player's (ghost) turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of ghost agent
                # Get the minimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost or pacman) turn now
                # Pass the new game state generated by ghost's `action` and the current alpha and beta values
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.alpha_beta(current_depth, agent_index + 1, next_game_state, alpha, beta)
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
                # Update the value of beta
                beta = min(beta, score)
                # Prune the tree if beta is less than alpha
                if beta < alpha:
                    break
        # If it is a leaf state with no successor states, return the value of evaluationFunction
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  # Return the best_action and best_score


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
        action, _ = self.expectimax(0, 0, gameState)  # Get the action and score for pacman (agent_index=0)
        return action  # Return the action to be done as per minimax algorithm

    def expectimax(self, current_depth, agent_index, gameState):
        '''
        Returns the best score for an agent using the expectimax algorithm. For max player (agent_index = 0), the best
        score is the maximum score among its successor states and for the min player (agent_index != 0), the best
        score is the average of all its successor states. Recursion ends if there are no successor states
        available or current_depth equals the max depth to be searched until.
        :param current_depth: the current depth of the tree (int)
        :param agent_index: index of the current agent (int)
        :param gameState: the current state of the game (GameState)
        :return: action, score
        '''
        # Roll over agent index and increase current depth if all agents have finished playing their turn in a move
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            current_depth += 1
        # Return the value of evaluationFunction if max depth is reached
        if current_depth == self.depth:
            return None, self.evaluationFunction(gameState)
        # Initialize best_score and best_action with None
        best_score, best_action = None, None
        if agent_index == 0:  # If it is max player's (pacman) turn
            for action in gameState.getLegalActions(agent_index):  # For each legal action of pacman
                # Get the expectimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost) turn now
                # Pass the new game state generated by pacman's `action`
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(current_depth, agent_index + 1, next_game_state)
                # Update the best score and action, if best score is None (not updated yet) or if current score is
                # better than the best score found so far
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
        else:  # If it is min player's (ghost) turn
            ghostActions = gameState.getLegalActions(agent_index)
            if len(ghostActions) is not 0:
                prob = 1.0 / len(ghostActions)
            for action in gameState.getLegalActions(agent_index):  # For each legal action of ghost agent
                # Get the expectimax score of successor
                # Increase agent_index by 1 as it will be next player's (ghost or pacman) turn now
                # Pass the new game state generated by ghost's `action`
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(current_depth, agent_index + 1, next_game_state)

                if best_score is None:
                    best_score = 0.0
                best_score += prob * score
                best_action = action
        # If it is a leaf state with no successor states, return the value of evaluationFunction
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score  # Return the best_action and best_score


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
        Evaluation function that INCREASES the evaluation when pacman:
            -eats food
            -is close to food
            -eats capsule
            -is close to capsule
            -is close to a scared ghost
            -wins (big increase)
        and DECREASES the evaluation when pacman:
            -is close to a non-scared ghost
            -does no progress
            -loses (big decrease)
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()  #initial evaluation is the current score so evaluation decreases when pacman does nothing

    pacmanPos = currentGameState.getPacmanPosition()
    foodLeft = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsulesLeft = currentGameState.getCapsules()
    
    if len(capsulesLeft) > 0:  #the closer to the closest capsule the better
        capsuleDistances = [manhattanDistance(capsule, pacmanPos) for capsule in capsulesLeft]
        score -= min(capsuleDistances)
    
    if len(foodLeft) > 0:  #the closer to the closest food the better
        foodDistances = [manhattanDistance(food, pacmanPos) for food in foodLeft]
        score -= min(foodDistances)
    
    if currentGameState.hasFood(pacmanPos[0], pacmanPos[1]):    # increase score if food is reached
        score += 50   
    if currentGameState.isLose():                               # huge decrease in score if state is loss
        score -= 9999    
    if currentGameState.isWin():                                # huge increase in score if state is loss
        score += 9999    
    for capsule in capsulesLeft:                                # increase score if capsule is reached
        if pacmanPos == capsule: 
            score += 100


    for ghost in ghostStates: 
        if ghost.scaredTimer > 0:   #if ghost is scared then the closer the better
            score += manhattanDistance(ghost.getPosition(), pacmanPos)
        else:   #if ghost not scared then the closest the worse
            score -= manhattanDistance(ghost.getPosition(), pacmanPos)

    score -= len(foodLeft)  #the more food left the worse

    return score


# Abbreviation
better = betterEvaluationFunction
