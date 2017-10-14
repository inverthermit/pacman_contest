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

def createTeam(firstIndex, secondIndex, isRed,#DefensiveReflexAgent OffensiveReflexAgent
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
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


  def getSuccessors(self, position, isChased):
      # TODO: Add the deadend configuration
      blockers = copy.copy(self.walls)
      if(isChased):
          blockers.extend(self.deadEndList)
      curOb = self.getCurrentObservation();
      if curOb.getAgentState(self.index).isPacman:
          enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
          defenders = [ele for ele in enemies if not ele.isPacman and ele.getPosition() != None and ele.scaredTimer <= 0]
          if len(defenders) > 0:
              defendersPos = [i.getPosition() for i in defenders]
              blockers.extend(defendersPos)
      if not curOb.getAgentState(self.index).isPacman:
          enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
          defenders = [ele for ele in enemies if not ele.isPacman and ele.getPosition() != None and ele.scaredTimer <= 0]
          if len(defenders) > 0:
              defendersPos = [i.getPosition() for i in defenders]
              blockers.extend(defendersPos)

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
            return self.heuristicSearch([self.start])
        # if len(goalList)==0:
        #     return 'North'

        return 'Stop'


  def getNiceClosestFood(self, gameState, defendFood=False, num=-1):
      position = self.getCurrentObservation().getAgentPosition(self.index)
      if defendFood:
          foodList = self.getFoodYouAreDefending(gameState).asList()
      else:
          curOb = self.getCurrentObservation();
          position = curOb.getAgentPosition(self.index)
          enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
          enemiesAtHome = [ele for ele in enemies if ele.isPacman and ele.getPosition() != None]
          if len(enemiesAtHome) > 0:
              dists = [self.getMazeDistance(position, ele.getPosition()) for ele in enemiesAtHome]
              for i in range(0, len(enemiesAtHome)):
                  if self.getMazeDistance(position, enemiesAtHome[i].getPosition()) == min(dists):
                      return [enemiesAtHome[i].getPosition()]
          foodList = self.getFood(gameState).asList()
      collection = util.PriorityQueue()
      for food in foodList:
          dis = self.getMazeDistance(position, food)
          collection.push((food), dis)
      result = []
      if(num<0):
          #for i in range(collection.count/4):
          for i in range(collection.count):
              result.append(collection.pop())
      else:
          if(collection.count<num):
              num = collection.count
          for i in range(num):
              result.append(collection.pop())
    #   print(len(result))
      return result

  def defenderBestPosition(self, gameState):
      curOb = self.getCurrentObservation();
      position = curOb.getAgentPosition(self.index)
      enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
      enemiesAtHome = [ele for ele in enemies if ele.isPacman and ele.getPosition() != None]
      if len(enemiesAtHome) > 0:
          dists = [self.getMazeDistance(position, ele.getPosition()) for ele in enemiesAtHome]
          for i in range(0, len(enemiesAtHome)):
              if self.getMazeDistance(position, enemiesAtHome[i].getPosition()) == min(dists):
                  return [enemiesAtHome[i].getPosition()]
      else:
          defFoodList = self.getNiceClosestFood(gameState, defendFood=True, num=3)
          ranPoint = random.choice(defFoodList)
          if position ==  ranPoint:
              defFoodList.remove(ranPoint)
              ranPoint = random.choice(defFoodList)
          return [ranPoint]


  def getGoals(self, gameState, isDefender):

      if not isDefender:
          return self.getNiceClosestFood(gameState)
      else:
          return self.defenderBestPosition(gameState)#defenderBestPosition

class OffensiveReflexAgent(ReflexCaptureAgent):
    isChased = False

    def chooseAction(self, gameState):

        foodList = self.getFood(gameState).asList()
        foodAte = self.foodNum - len(foodList)
        print foodAte
        curOb = self.getCurrentObservation()
        selfState = curOb.getAgentState(self.index)

        #Go back to home
        if len(foodList) == 2:
            return self.heuristicSearch([self.start])


        #already escape
        if self.isChased == True and not selfState.isPacman:
            self.isChased = False
            return self.heuristicSearch([self.start])



        #avoid defenders
        if selfState.isPacman:
            # get defenders position
            enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
            enemiesAtHome = [ele for ele in enemies if not ele.isPacman and ele.getPosition() != None and ele.scaredTimer <= 0]
            if len(enemiesAtHome) > 0:
                defendersPos = [i.getPosition() for i in enemiesAtHome]

                for pos in defendersPos:
                    distance = self.getMazeDistance(pos,selfState.getPosition()) - 2
                    if distance <= 2:
                        self.chaseByDefender = True
                        return self.heuristicSearch([self.start])

        # create a function for better go home
        if foodAte>= 5 and foodAte!=0 and selfState.isPacman:
            print('Go home!!!')
            # if food is near, eat food
            closestFood = self.getNiceClosestFood(gameState, defendFood=False, num=1)
            distance = self.getMazeDistance(closestFood[0],selfState.getPosition())
            # print('distance to the closest food:'+str(distance))
            if(distance == 1):
                return self.heuristicSearch(closestFood)
            return self.heuristicSearch([self.start])

        if not selfState.isPacman:
            self.foodNum = len(self.getFood(gameState).asList())

        return self.heuristicSearch(self.getGoals(gameState,False))



class DefensiveReflexAgent(ReflexCaptureAgent):
  def chooseAction(self, gameState):
    #   return 'Stop'
      return self.heuristicSearch(self.getGoals(gameState,True))
