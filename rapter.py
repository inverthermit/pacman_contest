# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

import distanceCalculator
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

######################
# ReflexCaptureAgent #
######################

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.numFood = foodLeft = len(self.getFood(gameState).asList())
    self.initialFoodNum = len(self.getFood(gameState).asList())

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    # print "************************Start*****************************************"
    # print "Agent Index:", self.index
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    # myTeam's state
    myTeam = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
    # once you have 9 points, go back to deposite your points
    if not myTeam[0].isPacman and not myTeam[1].isPacman and self.index == 0:
        self.numFood = foodLeft
    elif myTeam[0].isPacman and not myTeam[1].isPacman and self.index == 0:
        self.foodEaten = self.numFood - foodLeft
        if (self.foodEaten == round(self.initialFoodNum / 5)):
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start,pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction


    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    # print "current action: ", action
    # print "features: ", features
    # print "weights: ", weights
    # print "feature * weights: ", features * weights
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
  def getEuclideanDistance(self, position1, position2):
      "The Euclidean distance heuristic for a PositionSearchProblem"
      xy1 = position1
      xy2 = position2
      return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    # print successor
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    myPos = successor.getAgentState(self.index).getPosition()
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    #print "data", gameState.data

    # try to avoid the ghost from opponent ghost
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    #print "enemies", enemies[0].scaredTimer
    #print "enemies", enemies[1]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    if len(invaders) != 0:
        invaderPos = [a.getPosition() for a in invaders]
    ghostPos = [g.getPosition() for g in ghosts]

    # if enemy's pacman is close(within 1 distance) to my ghost(pacman)
    if not successor.getAgentState(self.index).isPacman and len(invaders) != 0:
        myPos = successor.getAgentState(self.index).getPosition()
        disToPac = min([self.getMazeDistance(myPos, pac) for pac in invaderPos])
        if disToPac <= 1:
            # print "catch ghost!"
            features['distanceToInv'] = disToPac

    # handle both ghosts are at the boarder and inifinte looping with each other

    if not successor.getAgentState(self.index).isPacman and len(ghosts) != 0:
        myPos = successor.getAgentState(self.index).getPosition()
        disToPac = min([self.getMazeDistance(myPos, g) for g in ghostPos])
        features['ghostDistance'] = disToPac
    elif len(ghosts) > 0:
        dists = [self.getEuclideanDistance(myPos, ghost.getPosition()) for ghost in ghosts]
        features['ghostDistance'] = min(dists)
        minDis = min(dists)
        index = 0
        ## handle when oppoent get scared
        for i in range(len(dists)):
            if minDis == dists[i]:
                index = i
        if enemies[index].scaredTimer != 0:
            # print "scared!"
            features['ghostDistance'] = 0
        else:
          # returns noise distance to each agents in current state.
          # print successor.getAgentDistances()
          # makeObservation
          #print help(successor.getDistanceProb), exit()
      # if ghosts around five step, do not try to look for food any more
          features['distanceToFood'] = 0
          # pacman do not want to get into shallow deadend
          if len(successor.getLegalActions(self.index)) == 2:
            features['deadend'] = 1
          else:
            features['deadend'] = 0
          # when pacman have four choices of action. complicate issue
          if len(gameState.getLegalActions(self.index)) == 4:
            successorActions = successor.getLegalActions(self.index)
            # successorSec is the sucessor of successor
            successorSec = [self.getSuccessor(successor, a) for a in successorActions]
            listofActions = [s.getLegalActions(self.index) for s in successorSec]
            #print "listofActions: ", listofActions
            deadEndList = [len(s.getLegalActions(self.index)) for s in successorSec]
            #print "corresponding length of all actions: ", deadEndList
            for numActions in deadEndList:
              if numActions == 2:
                features['deadend'] += 1
    if action == Directions.STOP: features['stop'] = 1

    return features


  def getWeights(self, gameState, action):
      return {'successorScore': 100, 'distanceToFood': -1, 'ghostDistance': 200, 'stop': -300, 'deadend': -200, 'distanceToInv': 50}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    # the line below prevent ghost become pacman, can lose the tie.
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      # for a in invaders:
      #     print "Ghost position", a.getPosition(), "MyPosition: ", myPos
      # features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
      # 'numInvaders' has not been used as feature.
    return {'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
