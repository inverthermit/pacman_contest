import util
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
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='AStarFoodSearchAgent', second='TANKAgent'):
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

class AStarFoodSearchAgent(CaptureAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.strtGameState = gameState
        self.powerTimer = 0
        self.distanceToHome = 0

    def chooseAction(self, gameState):
        print "****************"
        """
        Picks among the actions with the highest Q(s,a).
        """

       # if len(self.getFood(self.strtGameState).asList()) > len(self.getFood(gameState).asList()):




        allaction = gameState.getLegalActions(self.index)

        actions = []
        if self.getMyPos(gameState) in self.getCapsules( self.strtGameState):
            self.powerTimer = 40
            print "laoziniubile"

        #print "gengxinle"
        self.strtGameState = gameState

            # If powered, reduce power timer each itteration
        if self.powerTimer > 0:
            self.powerTimer -= 1

        foodLeft = len(self.getFood(gameState).asList())
        for a in allaction:
            if a != 'Stop':
                actions += [a]
        self.distanceToHome = self.disToHome(gameState)
        if self.powerTimer > self.distanceToHome + 5:
            print "powered left " + str(self.powerTimer)
            values = [(self.evaluate(gameState, a)) for a in actions]
            print "all action value " + str(values)
            minValue = min(values)
            bestActions = [a for a, v in zip(actions, values) if v == minValue and a != 'Stop']
            print "best action " + str(bestActions)

            # print bestActions
            return random.choice(bestActions)

        if self.powerTimer > 0:
            print "gohomewith power"
            print "powered left " + str(self.powerTimer)
            values = [(self.gohomeHeuristic(gameState, a)) for a in actions]
            print "all action value " + str(values)
            minValue = min(values)
            bestActions = [a for a, v in zip(actions, values) if v == minValue and a != 'Stop']
            print "best action " + str(bestActions)

            # print bestActions
            return random.choice(bestActions)
        actions = self.getPossibleAction(gameState,actions)



        if self.getScore(gameState) > 5:
            foodlimit = 2
        else:
            foodlimit = 1

        if foodLeft <= 2 or (gameState.getAgentState(self.index).numCarrying > foodlimit) or (gameState.getAgentState(self.index).numCarrying > 0 and self.enemyDist(gameState) < 3):
            print "gohome"
            print "all legal action " + str(actions)
            #values = [(self.gohomeHeuristic(gameState,a) + self.enemyHeuristic(gameState,a) )for a in actions]
            values = [(self.gohomeHeuristic(gameState,a) - self.getActionPayoff(gameState,a) )for a in actions]
            print "all action value " + str(values)
            minValue = min(values)
            bestActions = [a for a, v in zip(actions, values) if v == minValue and a != 'Stop']
            print "best action " + str(bestActions)
            return random.choice(bestActions)




        print "all legal action " + str(actions)
        # You can profile your evaluation time by uncommenting these lines
        # start = tixme.time()
        x = gameState.getWalls().width / 2
        #values = [(self.evaluate(gameState, a) + self.enemyHeuristic(gameState,a)) for a in actions]
        values = [(self.evaluate(gameState, a)- self.getActionPayoff(gameState,a)) for a in actions]
        print "all action value " + str(values)
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        minValue = min(values)
        bestActions = [a for a, v in zip(actions, values) if v == minValue and a != 'Stop']
        print "best action " + str(bestActions)


        #print bestActions
        return random.choice(bestActions)
    def enemyHeuristic(self,gameState,action):
        successor,cost = self.getSuccessor(gameState,action)
        enemyDistence = self.enemyDist(successor)
        enemy = self.getEnemyPos(successor)
        if enemyDistence == None or (not gameState.getAgentState(self.index).isPacman):
            return 0
        return max((5 - enemyDistence) * 10,0)
    def foodHeuristic(self, gameState,preGameState):
        """
        Your heuristic for the FoodSearchProblem goes here.

        This heuristic must be consistent to ensure correctness.  First, try to come
        up with an admissible heuristic; almost all admissible heuristics will be
        consistent as well.

        If using A* ever finds a solution that is worse uniform cost search finds,
        your heuristic is *not* consistent, and probably not admissible!  On the
        other hand, inadmissible or inconsistent heuristics may find optimal
        solutions, so be careful.

        The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
        (see game.py) of either True or False. You can call foodGrid.asList() to get
        a list of food coordinates instead.

        If you want access to info like walls, capsules, etc., you can query the
        problem.  For example, problem.walls gives you a Grid of where the walls
        are.

        If you want to *store* information to be reused in other calls to the
        heuristic, there is a dictionary called problem.heuristicInfo that you can
        use. For example, if you only want to count the walls once and store that
        value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
        Subsequent calls to this heuristic can access
        problem.heuristicInfo['wallCount']
    """


        position = self.getMyPos(gameState)
        foodGrid = self.getFood(preGameState).asList()
        nextGrid = self.getFood(gameState).asList()
        if len(nextGrid) - len(foodGrid) > 0:
            return -1000
        #distanceList = [0]
        i = 0
        mindistins = [(self.getMazeDistance(position, food)) for food in foodGrid]


        foodHeuristicValue = min(mindistins)
        #print foodHeuristicValu
        return foodHeuristicValue

        # Find which enemy is the closest

    def getPossibleAction(self,gameState,actions):
        possibleAction = []
        for a in actions:
            successor,cost = self.getSuccessor(gameState , a)
            if self.getMyPos(successor) not in self.getEnemyPoslist(gameState):
                possibleAction.append(a)
        if len(possibleAction) > 0:
            return possibleAction
        else :
            return [random.choice(actions)]

    def enemyDist(self, gameState):
            pos = self.getEnemyPos(gameState)
            minDist = None
            if len(pos) > 0:
                minDist = float('inf')
                myPos = self.getMyPos(gameState)
                for i, p in pos:
                    dist = self.getMazeDistance(p, myPos)
                    if dist < minDist:
                        minDist = dist
            return minDist
    def disToHome(self, gameState):
        locations = []
        self.atCenter = False
        x = gameState.getWalls().width / 2
        y = 0
        # 0 to x-1 and x to width
        if self.red:
            x = x - 1
        self.center = (x, y)
        maxHeight = gameState.getWalls().height

        # look for locations to move to that are not walls (favor top positions)
        for i in xrange(maxHeight - y):
            if not gameState.hasWall(x, y) and (x, y) not in self.getEnemyPoslist(gameState):
                locations.append((x, y))
            y = y + 1
        print 'VVVVVV'
        print self.distanceToHome

        locations += self.getCapsules(gameState)
        myPos = gameState.getAgentState(self.index).getPosition()

        minDist = float('inf')
        minPos = None

        for location in locations:
            dist = self.getMazeDistance(myPos, location)
            if dist <= minDist:
                minDist = dist
                minPos = location
        return minDist

    def disttoCenter(self, gameState,preGameState):
        locations = []
        self.atCenter = False
        x = gameState.getWalls().width / 2
        y = 0
        # 0 to x-1 and x to width
        if self.red:
            x = x - 1
        self.center = (x, y)
        maxHeight = gameState.getWalls().height

        # look for locations to move to that are not walls (favor top positions)
        for i in xrange(maxHeight - y):
            if not gameState.hasWall(x, y) and (x,y) not in self.getEnemyPoslist(gameState) :
                locations.append((x, y))
            y = y + 1
        print 'VVVVVV'
        print self.distanceToHome
        if self.powerTimer <=  self.distanceToHome + 5:
            locations += self.getCapsules(preGameState)

        #print locations
        myPos = gameState.getAgentState(self.index).getPosition()


        minDist = float('inf')
        minPos = None

        for location in locations:
            dist = self.getMazeDistance(myPos, location)
            if dist <= minDist:
                minDist = dist
                minPos = location
        if minPos in self.getCapsules(preGameState) :
            if self.getMazeDistance(myPos,minPos) <= 1:
                return minDist - 9900
            else:
                return minDist - 999
        print myPos
        print minDist
        print '^^^^^^'
        return minDist

    def gohomeHeuristic(self,gameState,action):
        successor,action = self.getSuccessor(gameState,action)

        return self.disttoCenter(successor,gameState)

    def getEnemyPos(self, gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            # Will need inference if None
            if pos != None:
                enemyPos.append((enemy, pos))
        return enemyPos

    def getEnemyPoslist(self, gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            # Will need inference if None
            if pos != None:
                enemyPos.append(pos)
        return enemyPos

    def getMyEnemyPos(self,gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            # Will need inference if None
            if pos != None and self.getMazeDistance(self.getMyPos(gameState),pos) <= 5:
                enemyPos.append((enemy, pos))
        return enemyPos

    def aStarSearch(self,gameState):
        """Search the node that has the lowest combined cost and heuristic first."""

        #print()
        fringe = util.PriorityQueue()
        fringe.push((self.getMyPos(gameState), [], 0), 0)
        explored = {}
        foodList = self.getFood(gameState).asList()
        while not fringe.isEmpty():
            node, actions, cost = fringe.pop()
            if node in foodList:
                return actions[0]

            if explored.__contains__(node):
                continue



            explored[node] = cost

            action  = gameState.getLegalActions(self.index)
            successor = self.getSuccessor(gameState, action)

            for coord, direction, steps in successor:
                new_actions = actions + [direction]
                newcost = cost + steps
                newHeuristic = newcost + self.foodHeuristic(successor)
                if not explored.__contains__(coord):

                    # explored.append(coord)
                    fringe.push((coord, new_actions, newcost), newHeuristic)
                elif explored.get(coord) > newcost:
                    fringe.push((coord, new_actions, newcost), newHeuristic)
                    explored[coord] = newcost

        return []

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return (successor.generateSuccessor(self.index, action),action)
        else:
            return (successor,action)

    def getActionPayoff(self, gameState, action):
        #print self.getEnemyPos(gameState)
        features = util.Counter()
        features['possibleEaten'] = 0
        features['minDist'] = 9999
        features['isPath'] = -1
        if self.enemyDist(gameState) < 2:
            features['possibleEaten'] = 1
        pacNextState, cost = self.getSuccessor(gameState, action)
        if len(self.getMyEnemyPos(gameState)) == 0  :
                print "safe1"
                return 1000
        if not (gameState.getAgentState(self.index).isPacman):
            if self.getMazeDistance(self.getMyPos(gameState), self.getMyPos(pacNextState)) < 3 \
                    and not (pacNextState.getAgentState(self.index).isPacman and self.enemyDist(pacNextState) < 3):
                print "safe2"
                return 1000
            else:
                return -20


        if self.red:
            centerW = (gameState.data.layout.width - 2) / 2
        else:
            centerW = (gameState.data.layout.width - 2) / 2 + 1
        availablePos = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centerW, i):
                availablePos.append((centerW, i))
        #print availablePos


        currentPos = self.getMyPos(gameState)
        #pacNextState,cost = self.getSuccessor(gameState, action)
        oppoAgent = self.getOpponents(gameState)
        #enemies = [gameState.getAgentState(i) for i in oppoAgent]
        enemies = self.getMyEnemyPos(gameState)

        #threatAgent = [a for a in enemies if (not a.isPacman) and (a.getPosition() != None)]
        #print "enemy " + str(enemies)
        ghostNextPos = []
        pacNextPos = []
        pacNextPos.append(pacNextState.getAgentState(self.index).getPosition())
        if len(enemies) >= 1 :

            for ghost, ghostPos in enemies:
                #ghost = threatAgent[0]
                #ghostState = gameState.getAgentState(ghost)
                #ghostPos = gameState.getAgentPosition(ghost.index)

                if not (gameState.getAgentState(ghost).isPacman) and not ( self.powerTimer > 10):#

                    ghostNextActions = gameState.getLegalActions(ghost)
                    ghostNextPos += [gameState.generateSuccessor(ghost, a).getAgentState(ghost).getPosition() for a in
                                    ghostNextActions]

                    ghostNextPos.append(ghostPos)
                    #pacNextPos = pacNextState.getAgentState(self.index).getPosition()
                    ghostNextPos.append(gameState.getInitialAgentPosition(self.index))

                    if pacNextPos[0] in ghostNextPos:
                        features['possibleEaten'] += 1
                    else:
                        features['possibleEaten'] += 0
                    distancer = distanceCalculator.Distancer(gameState.data.layout)
                    #print pacNextPos
                    #print str(ghostNextPos) + "***"
                    #print "**pacmanNextPOS:" + str(pacNextPos)
                    #print "**ghost(C+N)POS:" + str(ghostNextPos)
                    minDist = min([self.getMazeDistance(pacNextPos[0], a) for a in ghostNextPos])
                    #minDist = min([distancer.getDistanceOnGrid(pacNextPos, a) for a in ghostNextPos])

                    features['minDist'] = min(minDist, features['minDist'])

                    """
                    find path - not going back
                    """
            visited = ghostNextPos
            visited.append(currentPos)
            #print "pacmanCurrentPOS:" +str(currentPos)
            #print "pacmanNextPOS:" +str(pacNextPos)
            #print "ghost(C+N)POS:"+str(ghostNextPos)
            startPos = util.Stack()
            startPos.push(pacNextPos[0])
            startState = util.Stack()
            startState.push(pacNextState)

            path = False
            while not startPos.isEmpty():
                initState = startState.pop()
                actions = initState.getLegalActions(self.index)
                currentVisit = startPos.pop()
                if currentVisit in availablePos:
                    path = True
                    break

                visited.append(currentVisit)
                nextState = [self.getSuccessor(initState, a)[0] for a in actions]
                for state in nextState:
                    temp = state.getAgentState(self.index).getPosition()
                    if temp in visited:
                        continue
                    else:
                        startPos.push(temp)
                        startState.push(state)
            #print "isThereAPath:"+str(path)
            if path:
                features['isPath'] = 1
            else:
                features['isPath'] = -1

        else:
            if features['minDist']==9999:
                return 1000

        print features['isPath']
        print features['possibleEaten']
        print features['minDist']
        payoff = features['isPath'] * 100 + features['possibleEaten'] * (-10000) + features['minDist'] * 3
        #print str(payoff)+"=/n"+"Path:"+str(features['isPath'] * 100)+"\n"+"Eaten:"+str(features['possibleEaten'] * (-10))+"\n"+"Dist:"+str(features['minDist'] * 3) +"\n"
        return payoff

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """

        successor = gameState.generateSuccessor(self.index, action)

        if (self.getMyPos(successor) in self.getFood(gameState).asList()) and len(gameState.getLegalActions(self.index)) > 3 :
            print "chidaole "
            return -1000
        return self.foodHeuristic(successor,gameState)

    def getMyPos(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = tixme.time()
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
                dist = self.getMazeDistance(self.start, pos2)
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

class TANKAgent(ReflexCaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.availablePos = []  # positions to patrol around
        self.preFood = None  # a list of food from previous state
        self.prey = None  # position now going to (x, y)
        self.field = {}  # record the probability of going to these locations

    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()
        # calculate the center of width
        if self.red:
            centerW = (gameState.data.layout.width - 2)/2
        else:
            centerW = ((gameState.data.layout.width - 2)/2) + 1

        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(centerW, i):
                self.availablePos.append((centerW, i))

        # is available pos is too large, then make it less
        while len(self.availablePos) > (gameState.data.layout.height -2)/2:
            self.availablePos.pop(0)
            self.availablePos.pop(-1)

        self.hunt(gameState)

    def hunt(self, gameState):
        """
        This method calculate the distance between TANK and the foods it is defending,
        it will then be used to calculate probability
        :param gameState:
        :return:
        """
        # Get a list of foods in your side
        foodDefending = self.getFoodYouAreDefending(gameState).asList()
        sum = 0
        for pos in self.availablePos:
            closestFoodDefDis = 999
            for food in foodDefending:
                dis = self.getMazeDistance(pos, food)
                if dis < closestFoodDefDis:
                    closestFoodDefDis = dis
            if closestFoodDefDis == 0:
                closestFoodDefDis = 1

            self.field[pos] = 1.0 / float(closestFoodDefDis)
            sum += self.field[pos]
        if sum == 0:
            sum = 1
        for x in self.field.keys():
            self.field[x] = float(self.field[x]) / float(sum)

    def selectPrey(self):
        keylist = self.field.keys()
        x = random.choice(keylist)
        return x

    # Return the time remaining scared
    def getScaredTimer(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer

    def chooseAction(self, gameState):
        """
        This function act by using self.prey, which is a coordinate that TANK is chasing
        :param gameState:
        :return:
        """
        currentFood= self.getFoodYouAreDefending(gameState).asList()

        # If you find your food is eaten, then go to the food location
        if self.preFood and len(self.preFood) < len(currentFood):
            self.hunt(gameState)

        myPos = gameState.getAgentPosition(self.index)
        if myPos == self.prey:
            self.prey = None

        # This block handles the situation that has invaders
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)

        if len(invaders) != 0:
            # Go to closest know invader
            positions = [inv.getPosition() for inv in invaders]
            self.prey = min(positions, key = lambda x: self.getMazeDistance(myPos, x))
        elif self.preFood != None:

            # Go to the place where food was eaten if someone ate them
            foodEaten = []
            for food in self.preFood:
                if food not in currentFood:
                    foodEaten.append(food)
            # Some food is eaten
            if len(foodEaten) != 0:
                # index means the latest location of food eaten
                self.prey = foodEaten.pop(0)

        self.preFood = currentFood

        # No food was eaten recently, then patrol around these food
        if self.prey is None and len(currentFood) <= 4:
            loc = currentFood
            loc += self.getCapsulesYouAreDefending()
            self.prey = random.choice(loc)
        elif self.prey is None:
            self.prey = self.selectPrey()

        # Check if our TANKAgent is scared
        availableActions = gameState.getLegalActions(self.index)

        if self.getScaredTimer(gameState) > 1:
            sucInfo = []  # A list keeping act info and its related suc distance to prey
            shortestDist = 999
            for act in availableActions:
                suc = self.getSuccessor(gameState, act)
                disToPrey = self.getMazeDistance(suc.getAgentPosition(self.index), self.prey)
                if disToPrey >= 2:
                    if disToPrey < shortestDist:
                        shortestDist = disToPrey
                        sucInfo = []
                        sucInfo.append((act, disToPrey))
                    elif disToPrey == shortestDist:
                        sucInfo.append((act, disToPrey))
            # shortestDis = sucInfo.sort(key = operator.itemgetter(1))[0][1]
            # acts = filter(lambda x: x[1] == shortestDist, sucInfo)
            resTuple = random.choice(sucInfo)
            resAction = resTuple[0]
            #print 'neart to powered pacman, take the action: ', resAction
            return resAction

        # Choose actions that makes agent close to target
        legit = []
        values = []
        availableActions.remove(Directions.STOP)
        for act in availableActions:
            suc = self.getSuccessor(gameState, act)

            if not suc.getAgentState(self.index).isPacman:
                sucLoc = suc.getAgentPosition(self.index)
                legit.append(act)
                values.append(self.getMazeDistance(sucLoc, self.prey))
        best = min(values)
        ties = filter(lambda x: x[0] == best, zip(values, legit))
        resAction = random.choice(ties)[1]

        #print 'TANKAgent'
        #print resAction
        return resAction
