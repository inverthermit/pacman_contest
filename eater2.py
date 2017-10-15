# junTeam.py song2_upgrade
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
    self.selfBorder,self.enermyBorder = self.getBoundaries(gameState)
    self.isChased = False
    self.forceGoPoint = None
    self.myTeamIndexes = self.getTeam(gameState)
    self.teammateIndex = 0
    # print('Whole Team!!!!!!')
    # print(self.myTeamIndexes)
    for i in range(len(self.myTeamIndexes)):
        if(self.index != self.myTeamIndexes[i]):
            self.teammateIndex = self.myTeamIndexes[i]
    # print('Teamate:')
    # print(self.teammateIndex)



# TODO: Calculate the boundaries using the first 15 seconds
  def getBoundaries(self, gameState):
    grid = gameState.getWalls()
    width = grid.width
    height = grid.height
    selfSide = self.start[0]
    leftBorder = width/2-1
    rightBorder = width/2
    selfBorder = []
    enermyBorder = []
    selfBorderX = leftBorder
    enermyBorderX = rightBorder
    if(selfSide>rightBorder):
        selfBorderX = rightBorder
        enermyBorderX = leftBorder

    # TODO: Start to calcultate no wall dot
    for i in range(height):
        if(not grid[selfBorderX][i] and not grid[enermyBorderX][i]):
            selfBorder.append((selfBorderX, i))
            enermyBorder.append((enermyBorderX, i))
    # print('*******************Self Border')
    # print(selfBorder)
    # print('*******************Enermy Border')
    # print(enermyBorder)

    # print('width:')
    # print(grid.width)
    # print('height:')
    # print(grid.height)
    return selfBorder,enermyBorder

  def getSuccessors(self, position, isChased):
      # TODO: Add the deadend configuration
      blockers = copy.copy(self.walls)
    #   if(isChased):
    #       blockers.extend(self.deadEndList)
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
              expandList = copy.copy(defendersPos)
              for element in defendersPos:
                  eBlock = (element[0]-1, element[1])
                  wBlock = (element[0]+1, element[1])
                  sBlock = (element[0], element[1]-1)
                  nBlock = (element[0], element[1]+1)
                  neBlock = (element[0]+1, element[1]+1)
                  nwBlock = (element[0]-1, element[1]-1)
                  seBlock = (element[0]+1, element[1]-1)
                  swBlock = (element[0]-1, element[1]+1)
                  fourDirectionList = [eBlock,wBlock,sBlock,nBlock,neBlock,nwBlock,seBlock,swBlock]
                  for b in fourDirectionList:
                      if b not in self.selfBorder:
                          expandList.append(b)


              blockers.extend(expandList)

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

  def getNiceClosestFood(self, gameState, defendFood=False, num=-1, customPosition = None):
      position = self.getCurrentObservation().getAgentPosition(self.index)
      if customPosition != None:
          position = customPosition
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
                  dis2Enemy = self.getMazeDistance(position, enemiesAtHome[i].getPosition())
                  if dis2Enemy == min(dists) and dis2Enemy<=5:
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


  def getClosestBorder(self, gameState):
      position = self.getCurrentObservation().getAgentPosition(self.index)
      collection = util.PriorityQueue()
      for border in self.selfBorder:
          dis = self.getMazeDistance(position, border)
          collection.push((border), dis)
      if collection.count==0:
          return [self.start]
      else:
          return [collection.pop()]

  def gameTheoryCalculation(self, gameState):
      curOb = self.getCurrentObservation()
      enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
      defenders = [ele for ele in enemies if not ele.isPacman and ele.getPosition() != None and ele.scaredTimer <= 0]
      matrix = []
      if len(defenders) > 0:
          defendersPos = [i.getPosition() for i in defenders]
          for i, selfBorderPoint in enumerate(self.selfBorder):
              row = []
              for j, enemyBoderPoint in enumerate(self.enermyBorder):
                  selfBorderClosestFood = self.getNiceClosestFood(gameState,defendFood=False, num=1,customPosition = selfBorderPoint)[0]
                  selfValue = self.getMazeDistance(selfBorderClosestFood, selfBorderPoint)
                  enemyValue = self.getMazeDistance(enemyBoderPoint, selfBorderPoint)
                  row.append( (-selfValue + enemyValue,-enemyValue) )
              matrix.append(row)
      maxIndex = -1
      maxValue = -99999
      for i,value in enumerate(matrix):
          rowSum = 0
          for rowVal in matrix[i]:
              rowSum = rowSum + rowVal[0]
          if(rowSum > maxValue):
              maxValue = rowSum
              maxIndex = i
      if(maxIndex == -1):
          return None
      return self.selfBorder[maxIndex]


  def gameTheoryCalCurrentDefender(self, gameState):
      print('into gameTheoryCalCurrentDefender')
      curOb = self.getCurrentObservation()
      enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
      defenders = [ele for ele in enemies if not ele.isPacman and ele.getPosition() != None and ele.scaredTimer <= 0]
      row = []
      defendersPos = [i.getPosition() for i in defenders]
      distribution = util.Counter()
      for i, selfBorderPoint in enumerate(self.selfBorder):
          selfBorderClosestFood = self.getNiceClosestFood(gameState,defendFood=False, num=1,customPosition = selfBorderPoint)[0]
          selfValue = self.getMazeDistance(selfBorderClosestFood, selfBorderPoint)
          rowValue = 0
          if len(defenders) > 0:
              enemyValueList = [self.getMazeDistance(ele,selfBorderPoint) for ele in defendersPos]
              print('enemyValueList:')
              print(enemyValueList)
              rowValue = -selfValue + min(enemyValueList)
          else:
              rowValue = -selfValue
          row.append(rowValue)
          distribution[selfBorderPoint] = rowValue
    #   self.displayDistributionsOverPositions(distribution)
      print(row)
      maxIndex = row.index(max(row))
      return self.selfBorder[maxIndex]








class OffensiveReflexAgent(ReflexCaptureAgent):



    def chooseAction(self, gameState):
        curOb = self.getCurrentObservation()
        selfState = curOb.getAgentState(self.index)
        teammateState = curOb.getAgentState(self.teammateIndex)
        selfPos = curOb.getAgentPosition(self.index)
        teammatePos = curOb.getAgentPosition(self.teammateIndex)
        # initializing with random border
        # if selfPos == self.start:
        #     ranBorder = random.choice(self.selfBorder)
        #     self.forceGoPoint = random.choice([None, ranBorder])
        # force go to a point
        if(self.forceGoPoint != None):
            print('Force go to*********'+str(self.forceGoPoint))
            position = curOb.getAgentPosition(self.index)
            if(position == self.forceGoPoint):
                self.forceGoPoint = None
            else:
                return self.heuristicSearch([self.forceGoPoint])


        # Recognize the state that it is stuck
        # if two of my ghosts are at border and are too close and
        # there is at least one very close ghost, force go somewhere(Game Theory).
        # Use random first. ran = random.choice([0, len(selfBorder)-1])
        # print(self.getTeam(gameState))

        if(True):#selfPos in self.selfBorder and teammatePos in self.selfBorder
            dis = self.getMazeDistance(selfPos,teammatePos)
            if dis <= 3:
                enemies = [curOb.getAgentState(i) for i in self.getOpponents(curOb)]
                enemiesAtHome = [ele for ele in enemies if not ele.isPacman and ele.getPosition() != None and ele.scaredTimer <= 0]
                if len(enemiesAtHome) > 0:
                    defendersPos = [i.getPosition() for i in enemiesAtHome]
                    for pos in defendersPos:
                        distance = self.getMazeDistance(pos,selfState.getPosition()) - 2
                        if distance <= 2:
                            # print('********into set forceGoPoint')
                            self.forceGoPoint = random.choice(self.selfBorder)
                            # self.forceGoPoint = self.gameTheoryCalCurrentDefender(gameState)
                            # self.forceGoPoint = random.choice([self.gameTheoryCalculation(gameState),self.gameTheoryCalCurrentDefender(gameState)])

                            return self.heuristicSearch([self.forceGoPoint])



        foodList = self.getFood(gameState).asList()
        foodAte = self.foodNum - len(foodList)
        print foodAte

        # if capsule is near, eat capsule
        capsules = self.getCapsules(gameState)
        for capsule in capsules:
            # print(capsule)
            distance = self.getMazeDistance(capsule,selfState.getPosition())
            if(distance <= 3):
                return self.heuristicSearch([capsule])
            # return self.heuristicSearch(self.getClosestBorder(gameState))




        #Go back to home
        if len(foodList) <= 2:
            return self.heuristicSearch(self.getClosestBorder(gameState))

        #already escape
        if self.isChased == True and not selfState.isPacman:
            self.isChased = False
            return self.heuristicSearch(self.getClosestBorder(gameState))



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
                        self.isChased = True
                        return self.heuristicSearch(self.getClosestBorder(gameState))

        # # create a function for better go home
        # border = self.getClosestBorder(gameState)
        # distance = self.getMazeDistance(border[0],selfState.getPosition())
        # if distance>0 and (foodAte*2)/distance >= 1 and foodAte!=0 and selfState.isPacman:
        #     print('Go home!!!')
        #     # if food is near, eat food
        #     closestFood = self.getNiceClosestFood(gameState, defendFood=False, num=1)
        #     distance = self.getMazeDistance(closestFood[0],selfState.getPosition())
        #     # print('distance to the closest food:'+str(distance))
        #     if(distance == 1):
        #         return self.heuristicSearch(closestFood)
        #     return self.heuristicSearch(border)
        # getCapsules

        if selfState.isPacman and curOb.getAgentState(self.teammateIndex).isPacman and self.isChased==False and self.getMazeDistance(selfPos,teammatePos)<=1 and selfState.numCarrying>teammateState.numCarrying:
            self.forceGoPoint = self.getClosestBorder(gameState)[0]
            return self.heuristicSearch([self.forceGoPoint])

        if not selfState.isPacman:
            self.foodNum = len(self.getFood(gameState).asList())

        return self.heuristicSearch(self.getGoals(gameState,False))

class DefensiveReflexAgent(ReflexCaptureAgent):
  def chooseAction(self, gameState):
    #   return 'Stop'
      return self.heuristicSearch(self.getGoals(gameState,True))
