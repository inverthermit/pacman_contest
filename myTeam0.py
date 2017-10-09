# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util, sys
from game import Directions
from util import nearestPoint
from game import Actions
import copy


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """


    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.walls = gameState.getWalls().asList()
        self.totalFoodNum = len(self.getFood(gameState).asList())



    # def getAvoidSuccessors(self, curPos):
    #     successors = []
    #     curWalls = copy.copy(self.walls)
    #     for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
    #         x, y = curPos
    #         dx, dy = Actions.directionToVector(action)
    #         nextx, nexty = int(x + dx), int(y + dy)
    #         if (nextx, nexty) not in curWalls:
    #             nextState = (nextx, nexty)
    #             successors.append((nextState, action))
    #     return successors


    def getSuccessors(self, curPos, curState):
        successors = []

        curWalls = copy.copy(self.walls)

        if curState.getAgentState(self.index).isPacman:
            # get defenders position
            enemies = [curState.getAgentState(i) for i in self.getOpponents(curState)]
            defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
            if len(defenders) > 0:
                defendersPos = [i.getPosition() for i in defenders]

                # allDangerPos = []
                # for pos in defendersPos:
                    # dangerPath = self.aStarSearchForAvoid(pos)
                    # if len(dangerPath) > 3:
                    #     allDangerPos.extend([dangerPath[-1],dangerPath[-2],dangerPath[-3]])
                    # elif len(dangerPath) > 2:
                    #     allDangerPos.extend([dangerPath[-1],dangerPath[-2]])
                    # elif len(dangerPath) > 1:
                    #     allDangerPos.append(dangerPath[-1])
                    # allDangerPos.append(de)

                curWalls.extend(defendersPos)

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = curPos
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in curWalls:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    def attackRoaming(self, gameState):

        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)

        foodList = self.getFood(gameState).asList()

        dist = 999999

        foodDistanceList = []
        for food in foodList:
            foodDistanceList.append(self.getMazeDistance(cur_position,food))

        shortestFoodList =  sorted(foodDistanceList)
        meantEatList = []


        for i in range(0,len(shortestFoodList)):
            for j in foodList:
                if self.getMazeDistance(cur_position, j) == shortestFoodList[i]:
                    meantEatList.append(j)
                    foodList.remove(j)

        firstFive = []
        for i in range(0, len(meantEatList)/4):
            firstFive.append(meantEatList[i])

        return firstFive

    def defendRoaming(self, gameState):


        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)
        # get invaders position
        enemies = [currentState.getAgentState(i) for i in self.getOpponents(currentState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        if len(invaders) > 0:
            dists = [self.getMazeDistance(cur_position, a.getPosition()) for a in invaders]
            for i in range(0, len(invaders)):
                if self.getMazeDistance(cur_position, invaders[i].getPosition()) == min(dists):
                    return [invaders[i].getPosition()]

        else:
            foodList = self.getFoodYouAreDefending(gameState).asList()
            foodListX = []
            PrimDefandPosition = []
            # defPosition = random.choice(foodList)

            for x in range(0, len(foodList)): foodListX.append(foodList[x][0])

            sortedList = sorted(foodListX)

            smallestX = sortedList[0]
            smallestX2 = sortedList[1]
            smallestX3 = sortedList[3]

            for i in range(0,len(foodList)):

                if foodList[i][0] == smallestX or foodList[i][0] == smallestX2 or  foodList[i][0] == smallestX3:
                    PrimDefandPosition.append(foodList[i])



            randomPosition = random.choice(PrimDefandPosition)
            if cur_position ==  randomPosition:
                PrimDefandPosition.remove(randomPosition)
                randomPosition = random.choice(PrimDefandPosition)
            return [randomPosition]



    def getGoals(self, gameState, isDefender):

        if not isDefender:
            return self.attackRoaming(gameState)
        else:
            return self.defendRoaming(gameState)


    def aStarSearch(self, goals):

        """Search the node that has the lowest combined cost and heuristic first."""
        currentState = self.getCurrentObservation()
        cur_position = currentState.getAgentPosition(self.index)

        for goal in goals:

            prioQueue = util.PriorityQueue()
            prioQueue.push((cur_position, []), util.manhattanDistance(cur_position, goal))
            visitedNodes = []

            while (prioQueue.isEmpty() == False):

                cur_position1, wholeWay = prioQueue.pop()

                if cur_position1 in visitedNodes:
                    continue
                visitedNodes.append(cur_position1)

                if cur_position1 == goal:
                    return wholeWay[0]
                cur_succ = self.getSuccessors(cur_position1, currentState)
                for s in cur_succ:
                    cost = len(wholeWay + [s[1]]) + util.manhattanDistance(s[0], goal)
                    prioQueue.push((s[0], wholeWay + [s[1]]), cost)

        #cannot find way then go home
        if goals[0] != self.start:
            self.aStarSearch([self.start])
        #cannot find way go home then wait for die
        return 'Stop'


    # def aStarSearchForAvoid(self,goal):
    #     currentState = self.getCurrentObservation()
    #     cur_position = currentState.getAgentPosition(self.index)
    #
    #
    #     prioQueue = util.PriorityQueue()
    #     prioQueue.push((cur_position, []), util.manhattanDistance(cur_position, goal))
    #     visitedNodes = []
    #
    #     while (prioQueue.isEmpty() == False):
    #
    #         cur_position1, wholePos = prioQueue.pop()
    #
    #         if cur_position1 in visitedNodes:
    #             continue
    #         visitedNodes.append(cur_position1)
    #
    #         if cur_position1 == goal:
    #             return wholePos
    #         cur_succ = self.getAvoidSuccessors(cur_position1)
    #         for s in cur_succ:
    #             cost = len(wholePos + [s[0]]) + util.manhattanDistance(s[0], goal)
    #             prioQueue.push((s[0], wholePos + [s[0]]), cost)

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    chaseByDefender = False

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """

        foodAte = self.totalFoodNum - len(self.getFood(gameState).asList())
        print foodAte
        selfCurState = self.getCurrentObservation().getAgentState(self.index)
        curState = self.getCurrentObservation()

        #left 2 food and go home
        if len(self.getFood(gameState).asList()) == 2:
            return self.aStarSearch([self.start])


        #move to another position
        if self.chaseByDefender == True and not selfCurState.isPacman:
            self.chaseByDefender = False
            return self.aStarSearch([self.start])



        #avoid defenders
        if selfCurState.isPacman:
            # get defenders position
            enemies = [curState.getAgentState(i) for i in self.getOpponents(curState)]
            defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]

            if len(defenders) > 0:
                defendersPos = [i.getPosition() for i in defenders]

                for pos in defendersPos:
                    distance = self.getMazeDistance(pos,selfCurState.getPosition()) - 2
                    if distance <= 2:
                        self.chaseByDefender = True
                        return self.aStarSearch([self.start])


        if foodAte == 5 and selfCurState.isPacman:
            return self.aStarSearch([self.start])

        if not selfCurState.isPacman:
            self.totalFoodNum = len(self.getFood(gameState).asList())

        return self.aStarSearch(self.getGoals(gameState,False))


    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

    def getFeatures(self, gameState):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}





class DefensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

    def chooseAction(self, gameState):
        return self.aStarSearch(self.getGoals(gameState,True))
