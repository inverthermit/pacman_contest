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
import time
import random
import numpy as np
from game import Directions
from game import Actions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
			   first = 'MixedAgent', second = 'MixedAgent'):
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

  # Pass the communication instance when creating team,
  # so the two agents can communicate with each other.
  communication = Communication(firstIndex, secondIndex)
  return [eval(first)(firstIndex, communication), eval(second)(secondIndex, communication)]


##########
# Agents #
##########
class Communication:
  '''
  This class is implemented for communication within two agents. The same
  instance would be passed to the two agents. They can't communicate by
  changing and reading the information of the instance
  '''
  def __init__(self, firstIndex, secondIndex):
	self.config = util.Counter()
	self.config[firstIndex] = 'defender'
	self.config[secondIndex] = 'attacker'
	self.closedFood = util.Counter()
	self.closedFood[firstIndex] = 99999
	self.closedFood[secondIndex] = 99999
	self.coordAttakerChasing = None


class MixedAgent(CaptureAgent):
  '''
  This is a class which contains the logics of both attacker and defender.
  '''
  def __init__(self, index, communication):
	'''
	Some definition of instance variables
	'''
	CaptureAgent.__init__(self, index)
	self.communication = communication
	self.simulationResult = []
	self.epsilon = 1
	self.epsilonFinal = 0.3
	self.discount = 0.7
	self.depth = 5
	self.startDecay = 20
	self.weights = {}
	self.offensive = True
	self.lastAction = None
	self.weights = {
	 'minGhostDistances': 100,
	 'minCapsuleChasingDistance': -5,
	 '#ofLeftFoods': -100,
	 'score': 100,
	 '#ofChasingCapsules': -100,
	 'minFoodDistance': -3,
	 'distanceFromStart': -10
	}
	self.target = None
	self.lastObservedFood = None
	self.patrolDict = {}
	self.timeReverse = 0


  def registerInitialState(self, gameState):
	'''
	Some calculation in the first 15s, which can help the agents decide the
	actions quickly
	'''
	CaptureAgent.registerInitialState(self, gameState)
	self.lastAction = None
	self.offensive = True
	# self.distancer.getMazeDistances()
	# self.start = gameState.getInitialAgentPosition(self.index)
	if self.red:
	  self.midWidth = (gameState.data.layout.width - 2)/2
	else:
	  self.midWidth = ((gameState.data.layout.width - 2)/2) + 1
	self.midHeight = gameState.data.layout.height/2
	self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
	self.opponents = self.getOpponents(gameState)
	self.distancer.getMazeDistances()
	self.team = self.index
	for sameTeam in self.getTeam(gameState):
	  if sameTeam != self.index:
		self.team = sameTeam
	# Compute central fields without walls from map layout.
	# The defender will walk among these positions when no
	# opponents ghosts are observable to defend its territory.
	self.noWallSpots = []
	for i in range(1, gameState.data.layout.height - 1):
	  if not gameState.hasWall(self.midWidth, i):
		self.noWallSpots.append((self.midWidth, i))
	# Remove some positions.
	while len(self.noWallSpots) > (gameState.data.layout.height -2)/2:
	  self.noWallSpots.pop(0)
	  self.noWallSpots.pop(len(self.noWallSpots)-1)
	# Initialize the probabilities to each patrol point.
	self.distFoodToPatrol(gameState)
	self.deadEnd = list()
	self.DFSExplore(gameState, [])
	self.timeReverse = 0


  def DFSExplore(self, currentState, visited):
	'''
	A recursive function which can mark all the dead ends in the map based on DFS
	'''
	# Given the state, find all the successor state and the successor coordinate
	currentPosition = currentState.getAgentPosition(self.index)
	visited.append(currentPosition)
	legalActions = currentState.getLegalActions(self.index)
	legalActions.remove(Directions.STOP)
	successors = [currentState.generateSuccessor(self.index, action) for action in legalActions]
	successorPos = [succ.getAgentPosition(self.index) for succ in successors]

	# If there is only one successor, then this must be the dead end
	if len(successors) == 1:
	  self.deadEnd.append(currentPosition)

	# Doing DFS recursively on each successors which are not visited yet
	for succ in successors:
	  if not succ.getAgentPosition(self.index) in visited:
		self.DFSExplore(succ, visited)

	# If some state has more then one successors, but only one of them is not
	# marked as dead end, then mark that state as the dead end.
	if len(successors) > 1:
	  notDeadEnd = 0
	  for succPos in successorPos:
		if not succPos in self.deadEnd:
		  notDeadEnd += 1
	  if notDeadEnd == 1:
		self.deadEnd.append(currentPosition)


  def chooseOffensiveAction(self, gameState):
	'''
	This is implemented for the attacker, using this function,
	the attacker can choose which actions to take
	'''
	# When the defender has given the order to chase some coordinate,
	# the attacker will do that with highest priority when it is a
	# ghost, otherwise it will ignore it
	if not gameState.getAgentState(self.index).isPacman and self.communication.coordAttakerChasing != None:
	  coordinateChasing = self.communication.coordAttakerChasing
	  currentPosition = gameState.getAgentPosition(self.index)
	  minDist = self.getMazeDistance(currentPosition, coordinateChasing)
	  legalActions = gameState.getLegalActions(self.index)
	  actionTaken = Directions.STOP
	  for action in legalActions:
		x, y = currentPosition
		dx, dy = Actions.directionToVector(action)
		newCoor = (x + dx, y + dy)
		dist = self.getMazeDistance(newCoor, coordinateChasing)
		if dist < minDist:
		  actionTaken = action
		  minDist = dist
	  return actionTaken

	startTime = time.time()
	features = self.getFeatures(gameState)
	eatenFood = gameState.getAgentState(self.index).numCarrying
	# If the pacman has eaten some food, and the distance to its own territory is
	# closer to the distance to the closest food, or carries more than three food
	# then try to go home
	if (eatenFood > 0 and features['distanceFromStart'] * 1.0 / eatenFood < features['minFoodDistance'] and features['#ofLeftFoods'] > 2) or eatenFood >= 3:
	  self.offensive = False
	# If carries some food and close to some ghost, then try to go home
	elif eatenFood > 0 and features['minGhostDistances'] > -100 and features['minGhostDistances'] < -1 / 3.01:
	  self.offensive = False
	# Otherwise, be offensive
	elif features['minFoodDistance'] <= -1 / 2.01:
	  self.offensive = True
	else:
	  self.offensive = True

	# if gameState.getAgentPosition(self.index) in self.deadEnd:
	#   print 'I am in dead end'
	# print self.getFeatures(gameState)
	# print self.weights
	if not self.offensive:
	  legalActions = gameState.getLegalActions(self.index)
	  legalActions.remove(Directions.STOP)
	  successors = [gameState.generateSuccessor(self.index, action) for action in legalActions]
	  qVal = [self.evaluateState(successor) for successor in successors]
	  maxQ = -999999
	  bestAct = Directions.STOP
	  for index in range(len(successors)):
		currentPosition = gameState.getAgentPosition(self.index)
		nextPosition = successors[index].getAgentPosition(self.index)
		if features['minGhostDistances'] < -1 / 6.01 and nextPosition in self.deadEnd:
		#   print 'I am chased, ',self.getNextAction(gameState, successors[index]),' leads to the dead end'
		  continue
		action = self.getNextAction(gameState, successors[index])
		if action != None and qVal[index] > maxQ:
		  bestAct = action
		  maxQ = qVal[index]
	  return bestAct

	self.simulationResult = []
	# simulation
	numOfSimulation = 0
	while True:
	  # start to decay
	  if self.startDecay < numOfSimulation:
		self.epsilon = max(self.epsilonFinal, self.epsilon * 0.9)
	  numOfSimulation += 1
	  actions = gameState.getLegalActions(self.index)
	  # in the first stage, we assume that we do not want to stay
	  # in some place
	  actions.remove(Directions.STOP)
	  legalActions = []
	  if numOfSimulation == 1:
		for action in actions:
		  if not self.takeToEmptyAlley(gameState, action, self.depth):
			nextState = gameState.generateSuccessor(self.index, action)
			if self.getNextAction(gameState, nextState) != None:
			  legalActions.append(action)
	  # we have a list of lists now to record current results of simulation
	  # each element is a three-element list, which contains [state, totalValue,
	  # number of simulations so far]
	  for action in legalActions:
		successor = gameState.generateSuccessor(self.index, action)
		self.simulationResult.append([successor,0.0,0])
	  # Explore or Exploit
	  stateIndex = 0
	  if random.random() < self.epsilon:
		# randomly choose a successor
		stateIndex = random.choice(range(len(self.simulationResult)))
	  else:
		stateIndex = self.chooseBestStateIndex()
	  if len(self.simulationResult) == 0:
		return Directions.STOP
	  nextState, currentVal, num = self.simulationResult[stateIndex]
	  simulationPath = self.randomSimulation(nextState, self.depth)
	  updatedVal = self.evaluatePath(simulationPath)
	  self.simulationResult[stateIndex][1] = (currentVal * num + updatedVal) / (num + 1)
	  self.simulationResult[stateIndex][2] = num + 1

	  # limit the time we do simulation
	  if time.time() - startTime >= 0.85:
		break

	results = []
	for (state, val, num) in self.simulationResult:
	  action = self.getNextAction(gameState, state)
	  results.append([action, val, num])

	nextStateIndex = self.chooseBestStateIndex()
	actionTaken = self.getNextAction(gameState, self.simulationResult[nextStateIndex][0])
	return actionTaken

  def takeToEmptyAlley(self, gameState, action, depth):
	"""
	Verify if an action takes the agent to an alley within given depth with
	no pacdots.
	"""
	if depth == 0:
	  return False
	newState = gameState.generateSuccessor(self.index, action)
	previousEatenFood = gameState.getAgentState(self.index).numCarrying
	eatenFood = newState.getAgentState(self.index).numCarrying
	if previousEatenFood < eatenFood:
	  return False
	actions  = newState.getLegalActions(self.index)
	actions.remove(Directions.STOP)
	reversed_direction = Directions.REVERSE[newState.getAgentState(self.index).configuration.direction]
	if reversed_direction in actions:
	  actions.remove(reversed_direction)
	if len(actions) == 0:
	  return True
	for a in actions:
	  if not self.takeToEmptyAlley(newState, a, depth - 1):
		return False
	return True

  def getNextAction(self, state, nextState):
	x1, y1 = state.getAgentPosition(self.index)
	x2, y2 = nextState.getAgentPosition(self.index)
	dx = x2 - x1
	dy = y2 - y1
	if abs(dx) > 1 or abs(dy) > 1:
	  return None
	else:
	  return Actions.vectorToDirection((dx,dy))

  def chooseBestStateIndex(self):
	bestVal = -9999999
	bestIndex = -1
	for index in range(len(self.simulationResult)):
	  nextState, currentVal, num = self.simulationResult[index]
	  if num > 0 and currentVal > bestVal:
		bestVal = currentVal
		bestIndex = index
	return bestIndex

  def randomSimulation(self, currentState, depth):
	simulationPath = [(currentState,Directions.STOP)]
	while depth >= 0:
	  legalActions = currentState.getLegalActions(self.index)
	  legalActions.remove(Directions.STOP)
	  previousAction = random.choice(legalActions)
	  successor = currentState.generateSuccessor(self.index, previousAction)
	  simulationPath.append((successor, previousAction))
	  currentState = successor
	  depth -= 1
	return simulationPath

  def evaluatePath(self, simulationPath):
	lastState = simulationPath[-1][0]
	qVal = self.evaluateState(lastState)
	for index in range(len(simulationPath) - 1, 1, -1):
	  nextState = simulationPath[index][0]
	  previousState = simulationPath[index - 1][0]
	  reward = self.getReward(previousState, nextState)
	  qVal = self.discount * qVal + reward
	return qVal

  def getReward(self, previousState, nextState):
	'''
	1. get food: small positive reward(+2)
	2. get capsult: medium positive reward(+5)
	3. get back to own territory with food: big positive reward(+10 x #food)
	4. get eaten: big negative reward(-500)
	5. travel in enemy territory: small negative reward(-1)
	6. travel in own territory vertically: medium negative reward(-2)
	7. travel in own territory horizontally towards own: medium negative reward(-2)
	8. travel in own territory horizontally towards enemy: small negative reward(-1)
	9. stop: medium negative reward(-5)
	'''
	features = self.getFeatures(previousState)
	featureNextState = self.getFeatures(nextState)
	agentState = previousState.getAgentState(self.index)
	agentStateNextState = nextState.getAgentState(self.index)
	reward = 0
	if nextState.getAgentPosition(self.index) in self.getFood(previousState).asList():
	  reward += 2
	if nextState.getAgentPosition(self.index) in self.getCapsules(previousState):
	  reward += 5
	if agentState.numCarrying > 0 and agentStateNextState.numCarrying == 0:
	  if self.getScore(nextState) > self.getScore(previousState):
		reward += 10 * previousState.getAgentState(self.index).numCarrying
	if agentState.isPacman and agentStateNextState.isPacman:
	  reward -= 1
	action = self.getNextAction(previousState,nextState)
	if action == None: #dead
	  reward -= 500
	if not agentState.isPacman and not agentStateNextState.isPacman:
	  if action in [Directions.NORTH,Directions.SOUTH]:
		reward -= 2
	  elif featureNextState['distanceFromStart'] < features['distanceFromStart']:
		reward -= 1
	  else:
		reward -= 2

	return reward

  def getFeatures(self, gameState):
	# initialization, record all the features using a dictionary
	features = util.Counter()

	myPosition = gameState.getAgentPosition(self.index)
	targetFood = self.getFood(gameState).asList()

	# Get the closest distance to the middle of the board.
	distanceFromStart = 0
	if gameState.getAgentState(self.index).isPacman:
	  distanceFromStart = 1 + min([self.getMazeDistance(myPosition, (self.midWidth, i))
								   for i in range(gameState.data.layout.height)
								   if (self.midWidth, i) in self.legalPositions])

	# Getting the distances to the oppnent agents that are ghosts.
	ghostDistances = []
	for oppnent in self.opponents:
	  if not gameState.getAgentState(oppnent).isPacman:
		oppnentPosition = gameState.getAgentPosition(oppnent)
		if oppnentPosition != None:
		  ghostDistances.append(self.getMazeDistance(myPosition, oppnentPosition))

	# Get the minimum distance of any of the ghost distances.
	# If it is greater than 4, we do not care about it so make it 0.
	minGhostDistances = min(ghostDistances) if len(ghostDistances) else 0

	capsulesChasing = self.getCapsules(gameState)
	capsulesChasingDistances = [self.getMazeDistance(myPosition, capsule) for capsule in
									  capsulesChasing]
	minCapsuleChasingDistance = min(capsulesChasingDistances) if len(capsulesChasingDistances) else 0
	foodDistances = [self.getMazeDistance(myPosition, food) for food in targetFood]
	minFoodDistance = min(foodDistances) if len(foodDistances) else 0
	minGhostDistances = min(ghostDistances) if len(ghostDistances) else 0
	features['distanceFromStart'] = distanceFromStart
	features['minGhostDistances'] = -1.0 / (minGhostDistances + 0.01)
	features['score'] = self.getScore(gameState)
	features['#ofLeftFoods'] = len(targetFood)
	features['minFoodDistance'] = minFoodDistance
	features['#ofChasingCapsules'] = len(capsulesChasing)
	features['minCapsuleChasingDistance'] = minCapsuleChasingDistance

	return features

  def evaluateState(self, state):
	qValue = 0
	features = self.getFeatures(state)
	scaredTimes = [state.getAgentState(oppnent).scaredTimer for oppnent in self.opponents]
	if self.offensive:
	  qValue += features['score'] * self.weights['score']
	  qValue += features['#ofLeftFoods'] * self.weights['#ofLeftFoods']
	  qValue += features['minFoodDistance'] * self.weights['minFoodDistance']
	  qValue += features['#ofChasingCapsules'] * self.weights['#ofChasingCapsules']
	  qValue += features['minCapsuleChasingDistance'] * self.weights['minCapsuleChasingDistance']
	  if min(scaredTimes) >= 6 and features['minGhostDistances'] < -1 / 4.01:
		features['minGhostDistances'] = 0
	  if features['minGhostDistances'] >= -1 / 4.01:
		features['minGhostDistances'] = 0
	  qValue += features['minGhostDistances'] * self.weights['minGhostDistances']
	else:
	  if min(scaredTimes) >= 6 and features['minGhostDistances'] < -1 / 4.01:
		features['minGhostDistances'] = 0
	  qValue += features['distanceFromStart'] * self.weights['distanceFromStart']
	  qValue += features['minGhostDistances'] * self.weights['minGhostDistances']  * 2
	  qValue += features['score'] * self.weights['score']

	if features['minGhostDistances'] > -100:
	  if len(state.getLegalActions(self.index)) < 3:
		qValue -= 10000
	  elif len(state.getLegalActions(self.index)) < 4:
		qValue -= 5000

	return qValue

  def distFoodToPatrol(self, gameState):
	"""
	This method calculates the minimum distance from our patrol
	points to our foods. The inverse of this distance will
	be measured as how likely we are going to select the patrol
	point as target.
	"""
	food = self.getFoodYouAreDefending(gameState).asList()
	total = 0

	# Get the minimum distance from the food to
	# patrol points.
	for position in self.noWallSpots:
	  closestFoodDist = "+inf"
	  for foodPos in food:
		dist = self.getMazeDistance(position, foodPos)
		if dist < closestFoodDist:
		  closestFoodDist = dist
	  if closestFoodDist == 0:
		closestFoodDist = 1
	  self.patrolDict[position] = 1.0/float(closestFoodDist)
	  total += self.patrolDict[position]
	# Normalize
	if total == 0:
	  total = 1
	for x in self.patrolDict.keys():
	  self.patrolDict[x] = float(self.patrolDict[x])/float(total)

  def selectPatrolTarget(self):
	"""
	Select the target.
	"""
	rand = random.random()
	sum = 0.0
	for x in self.patrolDict.keys():
	  sum += self.patrolDict[x]
	  if rand < sum:
		return x

	# Implemente este metodo para controlar o agente (1s max).

  def chooseDefensiveAction(self, gameState):
	'''
	This is implemented for the defender, using this function,
	the defender can choose which actions to take
	'''
	self.communication.coordAttakerChasing = None
	currentPos = gameState.getAgentPosition(self.index)
	teammatePos = gameState.getAgentPosition(self.team)
	if not gameState.getAgentState(self.team).isPacman:
	  teammateHuntedbyGhost = False
	  for opponent in self.opponents:
		oppnentPosition = gameState.getAgentPosition(opponent)
		if oppnentPosition != None and self.getMazeDistance(teammatePos, oppnentPosition) <= 5:
		  teammateHuntedbyGhost = True
		  break
	  if self.communication.closedFood[self.index] < self.communication.closedFood[self.team] or teammateHuntedbyGhost:
		for opponent in self.opponents:
		  oppoPos = gameState.getAgentPosition(opponent)
		  if oppoPos != None and self.getMazeDistance(currentPos, oppoPos) <= 5:

			break
		else:
		  self.communication.config[self.index] = 'attacker'
		  self.communication.config[self.team] = 'defender'


	# if the teammate is a pacman?
	if not gameState.getAgentState(self.team).isPacman:
	  for oppnent in self.opponents:
		oppnentPosition = gameState.getAgentPosition(oppnent)
		if oppnentPosition != None and self.getMazeDistance(oppnentPosition, teammatePos) <= 5 and gameState.getAgentState(oppnent).isPacman:
		  self.communication.coordAttakerChasing = oppnentPosition
		if oppnentPosition != None and self.getMazeDistance(currentPos, teammatePos) > self.getMazeDistance(currentPos, oppnentPosition) and gameState.getAgentState(oppnent).isPacman:
		  self.communication.coordAttakerChasing = oppnentPosition


	if self.lastObservedFood and len(self.lastObservedFood) != len(self.getFoodYouAreDefending(gameState).asList()):
	  self.distFoodToPatrol(gameState)

	mypos = gameState.getAgentPosition(self.index)
	if mypos == self.target:
	  self.target = None

	x = self.getOpponents(gameState)
	enemies  = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
	invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
	if len(invaders) > 0:
	  positions = [agent.getPosition() for agent in invaders]
	  self.target = min(positions, key = lambda x: self.getMazeDistance(mypos, x))
	elif self.lastObservedFood != None:
	  eaten = set(self.lastObservedFood) - set(self.getFoodYouAreDefending(gameState).asList())
	  if len(eaten) > 0:
		eatenFood = eaten.pop()
		defendingFoodList = self.getFoodYouAreDefending(gameState).asList()
		nextTarget = defendingFoodList[0]
		minDist = self.getMazeDistance(nextTarget, eatenFood)
		for foodRemain in defendingFoodList:
		  dist = self.getMazeDistance(foodRemain, eatenFood)
		  if dist < minDist:
			minDist = dist
			nextTarget = foodRemain
		self.target = nextTarget

	# self.lastObservedFood = self.getFoodYouAreDefending(gameState).asList()

	if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
	  food = self.getFoodYouAreDefending(gameState).asList() \
		   + self.getCapsulesYouAreDefending(gameState)
	  self.target = random.choice(food)
	elif self.target == None:
	  self.target = self.selectPatrolTarget()

	actions = gameState.getLegalActions(self.index)
	goodActions = []
	fvalues = []
	for a in actions:
	  new_state = gameState.generateSuccessor(self.index, a)
	  if not new_state.getAgentState(self.index).isPacman and not a == Directions.STOP:
		newpos = new_state.getAgentPosition(self.index)
		goodActions.append(a)
		fvalues.append(self.getMazeDistance(newpos, self.target))

	# Randomly chooses between ties.
	if len(fvalues) == 0:
	  return random.choice(actions)
	best = min(fvalues)
	ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))
	return random.choice(ties)[1]

  def chooseAction(self, gameState):

	# if gameState.getAgentPosition(self.index) in self.deadEnd:
	#   print 'This is dead end'
	# else:
	#   print 'This is not dead end'
	# raw_input()
	actionTaken = None
	foodList = self.getFood(gameState).asList()
	minFood = foodList[0]
	minDist = 99999
	for food in foodList:
	  dist = self.getMazeDistance(food, gameState.getAgentPosition(self.index))
	  if dist < minDist:
		minDist = dist
	self.communication.closedFood[self.index] = minDist
	if self.communication.config[self.index] == 'defender':
	  actionTaken = self.chooseDefensiveAction(gameState)
	else:
	  actionTaken = self.chooseOffensiveAction(gameState)
	if self.lastAction == Actions.reverseDirection(actionTaken):
	  self.timeReverse += 1
	else:
	  self.timeReverse = 0
	if self.timeReverse > 5:
	  acitonTaken = random.choice(gameState.getLegalActions(self.index))
	  self.timeReverse = 0
	self.lastAction = actionTaken
	self.lastObservedFood = self.getFoodYouAreDefending(gameState).asList()
	return actionTaken


class DefensiveAgent(MixedAgent):
  def chooseAction(self, gameState):
	teammate = 0
	for t in self.team:
	  if t != self.index:
		teammate = t
	return self.chooseDefensiveAction(gameState)


class OffensiveAgent(MixedAgent):
  def chooseAction(self, gameState):
	team = None
	actionTaken = None
	for teammate in self.team:
	  if teammate != self.index:
		team = teammate
		break
	for oppnent in self.opponents:
	  oppnentPosition = gameState.getAgentPosition(oppnent)
	  if oppnentPosition == None:
		continue
	  opponentState = gameState.getAgentState(oppnent)
	  myPos = gameState.getAgentPosition(self.index)
	  teammatePos = gameState.getAgentPosition(team)
	  teammateState = gameState.getAgentState(team)
	  if self.getMazeDistance(myPos, oppnentPosition) < 4 and opponentState.isPacman:
		actionTaken = self.chooseDefensiveAction(gameState)
	  if not gameState.getAgentState(self.index).isPacman and self.getMazeDistance(myPos, oppnentPosition) < self.getMazeDistance(myPos, teammatePos) and opponentState.isPacman and not teammateState.isPacman:
		actionTaken = self.chooseDefensiveAction(gameState)
	actionTaken = self.chooseOffensiveAction(gameState)
	self.lastAction = actionTaken
	return actionTaken


class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
	"""
	This method handles the initial setup of the
	agent to populate useful fields (such as what team
	we're on).

	A distanceCalculator instance caches the maze distances
	between each pair of positions, so your agents can use:
	self.distancer.getDistance(p1, p2)

	IMPORTANT: This method may run for at most 15 seconds.
	"""

	'''
	Make sure you do not delete the following line. If you would like to
	use Manhattan distances instead of maze distances in order to save
	on initialization time, please take a look at
	CaptureAgent.registerInitialState in captureAgents.py.
	'''
	CaptureAgent.registerInitialState(self, gameState)

	'''
	Your initialization code goes here, if you need any.
	'''


  def chooseAction(self, gameState):
	"""
	Picks among actions randomly.
	"""
	actions = gameState.getLegalActions(self.index)

	'''
	You should change this in your own agent.
	'''

	return random.choice(actions)
