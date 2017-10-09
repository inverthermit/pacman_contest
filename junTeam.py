# junTeam.py
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions
import copy

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

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.foodNum = len(self.getFood(gameState).asList())
    self.walls = gameState.getWalls().asList()
    self.deadEndList = self.getDeadEnd(gameState)
    self.boundaries = self.getBoundaries(gameState)


# TODO: Calculate the deadend using the first 15 seconds
  def getDeadEnd(self, gameState):
    return []

# TODO: Calculate the boundaries using the first 15 seconds
  def getBoundaries(self, gameState):
    return []

  # def chooseAction(self, gameState):
  #   """
  #   Picks among the actions with the highest Q(s,a).
  #   """
  #   actions = gameState.getLegalActions(self.index)
  #
  #   # You can profile your evaluation time by uncommenting these lines
  #   # start = time.time()
  #   values = [self.evaluate(gameState, a) for a in actions]
  #   # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
  #
  #   maxValue = max(values)
  #   bestActions = [a for a, v in zip(actions, values) if v == maxValue]
  #
  #   foodLeft = len(self.getFood(gameState).asList())
  #
  #   if foodLeft <= 2:
  #     bestDist = 9999
  #     for action in actions:
  #       successor = self.getSuccessor(gameState, action)
  #       pos2 = successor.getAgentPosition(self.index)
  #       dist = self.getMazeDistance(self.start,pos2)
  #       if dist < bestDist:
  #         bestAction = action
  #         bestDist = dist
  #     return bestAction
  #
  #   return random.choice(bestActions)

  def getSuccessors(self, position, isChased):
      # TODO: Add the deadend configuration
      blockers = copy.copy(self.walls)
      if(isChased):
          blockers.extend(self.deadEndList)
      successors = []
      for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
          x,y = position
          dx, dy = Actions.directionToVector(action)
          nextx, nexty = int(x + dx), int(y + dy)
          if (nextx, nexty) not in blockers:
              nextState = (nextx, nexty)
              successors.append((nextState, action))
      return successors

  def heuristicSearch(self, goalList):
        position = self.getCurrentObservation().getAgentPosition(self.index)
        def heuristic(p1,p2):
            return util.manhattanDistance(p1, p2)
        for point in goalList:
            collection = util.PriorityQueue()
            visited = []
            dis = heuristic(position, point)
            collection.push((position, []), dis)
            while not collection.isEmpty():
                tmpPoint, path = collection.pop()
                if(tmpPoint in visited):
                    continue
                visited.append(tmpPoint)
                if(point == tmpPoint):
                    if(len(path)==0):
                        continue
                    return path[0]
                successors = self.getSuccessors(tmpPoint, point)
                for element in successors:
                    fn = len(path + [element[1]]) + heuristic(element[0], point)
                    collection.push((element[0], path+[element[1]]), fn)
        if len(goalList)>0 and goalList[0]!=self.start:
            return self.heuristicSearch(position, [[self.start]])
        # if len(goalList)==0:
        #     return 'North'

        return 'Stop'


  def getNClosestFood(self, gameState):
      position = self.getCurrentObservation().getAgentPosition(self.index)
      foodList = self.getFood(gameState).asList()



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
          return self.attackRoaming(gameState)

class OffensiveReflexAgent(ReflexCaptureAgent):
    chaseByDefender = False

    def chooseAction(self, gameState):

        foodAte = self.foodNum - len(self.getFood(gameState).asList())
        # print foodAte
        selfCurState = self.getCurrentObservation().getAgentState(self.index)
        curState = self.getCurrentObservation()

        #left 2 food and go home
        if len(self.getFood(gameState).asList()) == 2:
            return self.heuristicSearch([self.start])


        #move to another position
        if self.chaseByDefender == True and not selfCurState.isPacman:
            self.chaseByDefender = False
            return self.heuristicSearch([self.start])



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
                        return self.heuristicSearch([self.start])


        if foodAte == 5 and selfCurState.isPacman:
            return self.heuristicSearch([self.start])

        if not selfCurState.isPacman:
            self.totalFoodNum = len(self.getFood(gameState).asList())

        return self.heuristicSearch(self.getGoals(gameState,False))
  # isBeingChased = False


class DefensiveReflexAgent(ReflexCaptureAgent):
  def chooseAction(self, gameState):
    #   return 'Stop'
      return self.heuristicSearch(self.getGoals(gameState,True))
