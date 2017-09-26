# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
import keyboardAgents
import game
from util import nearestPoint
import os


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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]



class ReflexCaptureAgent(CaptureAgent):
  def __init__( self, index, timeForComputing = .1 ):
    CaptureAgent.__init__( self, index, timeForComputing)
    self.visibleAgents = []

  def createPDDLobjects(self):
    result = '';

    """
    FILL THE CODE TO GENERATE PDDL OBJECTS
    """
    return result

  def createPDDLfluents(self):
    result = ''

    """
    FILL THE CODE TO GENERATE PDDL PREDICATES
    """


    return result

  def createPDDLgoal( self ):
    result = ''

    """
    FILL THE CODE TO GENERATE PDDL GOAL
    """
    return result

  def generatePDDLproblem(self):
	"""convierte un escenario en un problema de strips"""
        cd = os.path.dirname(os.path.abspath(__file__))
	f = open("%s/problem%d.pddl"%(cd,self.index),"w");
	lines = list();
	lines.append("(define (problem strips-log-x-1)\n");
   	lines.append("   (:domain pacman-strips)\n");
   	lines.append("   (:objects \n");
	lines.append( self.createPDDLobjects() + "\n");
	lines.append(")\n");
	lines.append("   (:init \n");
	lines.append("   ;primero objetos \n");
	lines.append( self.createPDDLfluents() + "\n");

        lines.append(")\n");
        lines.append("   (:goal \n");
        lines.append("	 ( and  \n");
        lines.append( self.createPDDLgoal() + "\n");
        lines.append("   ))\n");
        lines.append(")\n");

	f.writelines(lines);
	f.close();


  def runPlanner( self ):
	cd = os.path.dirname(os.path.abspath(__file__))
	os.system("%s/ff  -o %s/domain.pddl -f %s/problem%d.pddl > %s/solution%d.txt"
                %(cd,cd,cd,self.index,cd,self.index) );

  def parseSolution( self ):
    cd = os.path.dirname(os.path.abspath(__file__))
    f = open("%s/solution%d.txt"%(cd,self.index),"r");
    lines = f.readlines();
    f.close();

    for line in lines:
      pos_exec = line.find("0: "); #First action in solution file
      if pos_exec != -1:
        command = line[pos_exec:];
        command_splitted = command.split(' ')

        x = int(command_splitted[3].split('_')[1])
        y = int(command_splitted[3].split('_')[2])

        return (x,y)

      #
      # Empty Plan, Use STOP action, return current Position
      #
      if line.find("ff: goal can be simplified to TRUE. The empty plan solves it") != -1:
        return  self.getCurrentObservation().getAgentPosition( self.index )

  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)

    bestAction = 'Stop'


    """
    RUN PLANNER
    """
    self.generatePDDLproblem()
    self.runPlanner()
    (newx,newy) = self.parseSolution()

    for a in actions:
      succ = self.getSuccessor(gameState, a)

      """
      SELECT FIRST ACTION OF THE PLAN
      """
      if succ.getAgentPosition( self.index ) == (newx, newy):
        bestAction = a
        print self.index, bestAction, self.getCurrentObservation().getAgentPosition( self.index ) ,(newx,newy)

    return bestAction

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
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
      print action, features, successor
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

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

  """
  def generatePDDLproblem(self):


    YOU CAN OVERWRITE THE PDDL PROBLEM GENERATOR, AND HAVE ONE SPECIFIC FOR
    DEFENSIVE AND ONE SPECIFIC FOR OFFENSIVE
  """
