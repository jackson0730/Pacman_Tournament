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
from game import Directions, Actions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'HeuristicAgentAdvance2', second = 'HeuristicAgentAdvance2'):
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

##########
# Agents #
##########

inference = {}
enemyScaredTime = {}
class CaptureAgentWithTools(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.teamMate = [i for i in self.getTeam(gameState) if not i == self.index][0]
        self.w = gameState.getWalls().width // 2 - 1
        self.redW = self.w
        self.blueW = self.w + 1
        self.h = gameState.getWalls().height
        # self.leftBoundary = [(self.w, h) for h in range(self.h) if not gameState.getWalls()[self.w][h]]
        # self.rightBoundary = [(self.w+1, h) for h in range(self.h) if not gameState.getWalls()[self.w+1][h]]
        self.connectBoundary = [((self.w, h),(self.w+1, h)) for h in range(self.h) if not gameState.getWalls()[self.w][h] and not gameState.getWalls()[self.w+1][h]]
        self.leftBoundary = [l for (l,r) in self.connectBoundary]
        self.rightBoundary = [r for (l,r) in self.connectBoundary]
        if self.red:
            self.homeBoundary = self.leftBoundary
            self.enemyBoundary = self.rightBoundary
        else:
            self.homeBoundary = self.rightBoundary
            self.enemyBoundary = self.leftBoundary
        self.DistanceToHome = DistanceToHome(gameState.getWalls(), self.getMazeDistance)
        self.hutongMap = HuTong(gameState.getWalls())
        for i in self.getOpponents(gameState):
            inference[i] = ExactInference(i, gameState)
            inference[i].initialize()
            enemyScaredTime[i] = 0

        self.otherRegister()

    def otherRegister(self):
        pass

    def chooseAction(self, gameState):
        try:
        # if True:
            self.updateBeliefs(gameState)
            self.beliefGameState = self.generateBeliefGameState(gameState)
            action = self._chooseAction(gameState)

            self.currentCapsule = self.getCapsules(gameState)
            self.successor = gameState.generateSuccessor(self.index, action)
            self.nextState = self.successor.getAgentState(self.index)
            self.nextPos = self.nextState.getPosition()
            if self.nextPos in self.currentCapsule:
                for i in enemyScaredTime:
                    enemyScaredTime[i] = 40
            return action
        except:
            print("chooseAction ERROR")
            actions = gameState.getLegalActions(self.index)
            return random.choice(actions)

    def _chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

    def updateBeliefs(self, gameState):
        try:
            if self.index == 0:
                if not self.getPreviousObservation() == None:
                    inference[3].elapseTime(gameState)
                    if enemyScaredTime[3] > 0:
                        enemyScaredTime[3] += -1
            else:
                inference[self.index - 1].elapseTime(gameState)
                if enemyScaredTime[self.index - 1] > 0:
                    enemyScaredTime[self.index - 1] += -1
            for i in inference:
                inference[i].observe(self.index, gameState)
                if not gameState.getAgentPosition(self.index) == None:
                    enemyScaredTime[self.index] = gameState.getAgentState(self.index).scaredTimer
            # self.displayDistributionsOverPositions(inference[1].getBeliefDistribution())
            # self.debugDraw(inference[1].getBeliefDistribution().argMax(), (1,0,0), True)
            # self.debugDraw(inference[3].getBeliefDistribution().argMax(), (0,1,0))
        except:
            print("updateBeliefs ERROR")
            for i in inference:
                inference[i].initialize()

    def generateBeliefGameState(self, gameState):
        beliefGameState = gameState.deepCopy()
        for enemy in self.getOpponents(gameState):
            if gameState.getAgentPosition(enemy) == None:
                conf = game.Configuration(inference[enemy].getBeliefDistribution().argMax(), game.Directions.STOP)
                beliefGameState.data.agentStates[enemy] = game.AgentState(conf, False)
                beliefGameState.data.agentStates[enemy].scaredTimer = enemyScaredTime[enemy]
        return beliefGameState

class TestHeuristicAgent(CaptureAgentWithTools):
    def _chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, action) for action in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

class TestHeuristicAgentO(TestHeuristicAgent):
    def evaluate(self, gameState, action):
        score = 0
        successor = gameState.generateSuccessor(self.index, action)
        myState = successor.getAgentState(self.index)
        myPos = successor.getAgentState(self.index).getPosition()

        foodList = self.getFood(successor).asList()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.scaredTimer <= 0 and a.getPosition() != None and self.getMazeDistance(myPos, a.getPosition()) <= 6]
        if len(defenders) > 0:
            score += -3000
            minDefendersDistance = min([self.getMazeDistance(myPos, defender.getPosition()) for defender in defenders])
            score += 10 * minDefendersDistance
            if minDefendersDistance == 1:
                score += -10000
            score += -20 * self.hutongMap.value(myPos)
            safeFood = [self.getMazeDistance(myPos, food) for food in foodList if self.hutongMap.value(food) == 0]
            score += -200 * len(safeFood)
            if not safeFood == []:
                minSafeFoodDistance = min(safeFood)
                score += -minSafeFoodDistance
            score += -0.7 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
        elif len(foodList) > 0:
            score += -100 * len(foodList)
            minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            score += -minFoodDistance
            score += -0.14 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
        score += -200 * len([a for a in enemies if a.isPacman or a.scaredTimer > 0 and a.getPosition() != None])
        if action == Directions.STOP:
            score += -10
        score += -0.0001 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
        return score

class HeuristicAgentAdvance2(TestHeuristicAgent):
    def evaluate(self, gameState, action):
        self.currentState = gameState.getAgentState(self.index)
        self.currentPos = self.currentState.getPosition()
        self.successor = self.beliefGameState.generateSuccessor(self.index, action)
        self.nextState = self.successor.getAgentState(self.index)
        self.nextPos = self.nextState.getPosition()
        self.currentFood = self.getFood(gameState).asList()
        self.nextFood = self.getFood(self.successor).asList()
        self.currentCapsule = self.getCapsules(gameState)
        self.myCapsule = self.getCapsulesYouAreDefending(gameState)
        self.myFood = self.getFoodYouAreDefending(gameState).asList()

        self.teamMateState = gameState.getAgentState(self.teamMate)
        self.teamMatePos = self.teamMateState.getPosition()

        self.enemies = [self.beliefGameState.getAgentState(i) for i in self.getOpponents(gameState)]
        if self.red:
            self.invaders = [i for i in self.enemies if i.getPosition()[0] <= self.redW]
            self.defenders = [i for i in self.enemies if i.getPosition()[0] >= self.blueW]
        else:
            self.invaders = [i for i in self.enemies if i.getPosition()[0] >= self.blueW]
            self.defenders = [i for i in self.enemies if i.getPosition()[0] <= self.redW]

        if len(self.currentFood) == 0:
            return self.offence(gameState, action)

        minFoodDistanceMe = min([self.getMazeDistance(self.currentPos, food) for food in self.currentFood])
        minFoodDistanceTeam = min([self.getMazeDistance(self.teamMatePos, food) for food in self.currentFood])
        if minFoodDistanceMe <= minFoodDistanceTeam:
            return self.offence(gameState, action)
        else:
            return self.offence(gameState, action, first = False)

    def offence(self, gameState, action, first = True):
        if len(self.currentFood) <= 2 and self.currentState.numCarrying == 0:
            return self.defence(gameState, action)
        if not self.currentState.isPacman:
            if self.enemies[0].getPosition() in self.enemyBoundary and self.getMazeDistance(self.currentPos, self.enemies[0].getPosition()) <= 3:
                return self.defence(gameState, action)
            if self.enemies[1].getPosition() in self.enemyBoundary and self.getMazeDistance(self.currentPos, self.enemies[1].getPosition()) <= 3:
                return self.defence(gameState, action)
        if not self.currentState.isPacman or (self.currentPos in self.enemyBoundary and self.currentState.numCarrying == 0):
            mainRoadFood = [food for food in self.currentFood if self.hutongMap.value(food) == 0]
            if len(self.currentCapsule) == 0 and len(self.currentFood) <= 8:
                return self.defence(gameState, action)
            if len(self.currentCapsule) == 0 and not first:
                return self.defence(gameState, action)

        score = 0
        if self.dead():
            score += -10000
        if self.currentPos in self.enemyBoundary and self.currentState.numCarrying > 0 and self.nextPos in self.homeBoundary:
            score += 5000
        if not self.currentState.isPacman:
            disTeamA = max([self.getMazeDistance(self.teamMatePos, self.enemies[0].getPosition()), self.enemies[0].scaredTimer])
            disTeamB = max([self.getMazeDistance(self.teamMatePos, self.enemies[1].getPosition()), self.enemies[1].scaredTimer])
            currentSafeExitTeamA = self.safe(self.teamMateState, self.enemies[0])
            currentSafeCapsuleTeamA = self.safe(self.teamMateState, self.enemies[0], self.currentCapsule)
            currentSafeExitTeamB = self.safe(self.teamMateState, self.enemies[1])
            currentSafeCapsuleTeamB = self.safe(self.teamMateState, self.enemies[1], self.currentCapsule)
            if self.teamMateState.isPacman and len(self.currentCapsule) > 0 and ((disTeamA < 5 and len(currentSafeExitTeamA + currentSafeCapsuleTeamA) == 0) or (disTeamB < 5 and len(currentSafeExitTeamB + currentSafeCapsuleTeamB) == 0)):
                minCapsuleDistance = min([self.getMazeDistance(self.nextPos, capsule) for capsule in self.currentCapsule])
                score += -minCapsuleDistance
            elif len(self.currentFood) > 2:
                if first:
                    minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in self.currentFood])
                    score += - minFoodDistance
                    if self.currentPos in self.homeBoundary and self.teamMatePos in self.homeBoundary and self.getMazeDistance(self.teamMatePos, self.currentPos) < 5:
                        score += self.getMazeDistance(self.teamMatePos, self.nextPos)
                else:
                    myFood = [food for food in self.currentFood if self.getMazeDistance(self.currentPos, food) < self.getMazeDistance(self.teamMatePos, food)]
                    if len(myFood) == 0:
                        teamFood = self.greedyFoodList(self.teamMatePos, self.currentFood)
                        myFood = teamFood[int(-len(teamFood)/2):-1]
                    if len(myFood) > 0:
                        minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in myFood])
                        score += - minFoodDistance
                        if self.currentPos in self.homeBoundary and self.teamMatePos in self.homeBoundary and self.getMazeDistance(self.teamMatePos, self.currentPos) < 5:
                            score += self.getMazeDistance(self.teamMatePos, self.nextPos)
                    else:
                        return self.defence(gameState, action)
        else:
            currentSafeExitA = self.safe(self.currentState, self.enemies[0])
            currentSafeCapsuleA = self.safe(self.currentState, self.enemies[0], self.currentCapsule)
            currentSafeExitB = self.safe(self.currentState, self.enemies[1])
            currentSafeCapsuleB = self.safe(self.currentState, self.enemies[1], self.currentCapsule)
            nextSafeExitA = self.safe(self.nextState, self.enemies[0], first = False)
            nextSafeCapsuleA = self.safe(self.nextState, self.enemies[0], self.currentCapsule, first = False)
            nextSafeExitB = self.safe(self.nextState, self.enemies[1], first = False)
            nextSafeCapsuleB = self.safe(self.nextState, self.enemies[1], self.currentCapsule, first = False)

            disA = max([self.getMazeDistance(self.currentPos, self.enemies[0].getPosition()), self.enemies[0].scaredTimer])
            disB = max([self.getMazeDistance(self.currentPos, self.enemies[1].getPosition()), self.enemies[1].scaredTimer])
            # if abs(self.enemies[0].getPosition()[0] - self.w) > 3 and abs(self.enemies[0].getPosition()[0] - self.blueW) > 3 and self.enemies[0].isPacman:
            #     currentSafeExitA = self.enemyBoundary
            #     currentSafeCapsuleA = self.currentCapsule
            # if abs(self.enemies[1].getPosition()[0] - self.redW) > 3 and abs(self.enemies[1].getPosition()[0] - self.blueW) > 3 and self.enemies[1].isPacman:
            #     currentSafeExitB = self.enemyBoundary
            #     currentSafeCapsuleB = self.currentCapsule

            if disA >= 15 and disB >= 15:
                currentSafeList = self.enemyBoundary + self.currentCapsule
                nextSafeList = self.enemyBoundary + self.currentCapsule
            elif disA < 15 and disB >= 15:
                currentSafeList = currentSafeExitA + currentSafeCapsuleA
                nextSafeList = nextSafeExitA + nextSafeCapsuleA
            elif disA >= 15 and disB < 15:
                currentSafeList = currentSafeExitB + currentSafeCapsuleB
                nextSafeList = nextSafeExitB + nextSafeCapsuleB
            elif disA < 8 and disB < 8:
                currentSafeList = [p for p in (currentSafeExitA + currentSafeCapsuleA) if p in (currentSafeExitB + currentSafeCapsuleB)]
                nextSafeList = [p for p in (nextSafeExitA + nextSafeCapsuleA) if p in (nextSafeExitB + nextSafeCapsuleB)]
            elif disA < disB:
                currentSafeList = currentSafeExitA + currentSafeCapsuleA
                nextSafeList = nextSafeExitA + nextSafeCapsuleA
            else:
                currentSafeList = currentSafeExitA + currentSafeCapsuleA
                nextSafeList = nextSafeExitA + nextSafeCapsuleA
            if len(currentSafeList) == 0 and len(nextSafeList) == 0:
                if disA < disB:
                    targets = (currentSafeExitA + currentSafeCapsuleA)
                else:
                    targets = (currentSafeExitB + currentSafeCapsuleB)
                if len(targets) > 0:
                    minTargetDistance = min([self.getMazeDistance(self.nextPos, target) for target in targets])
                    score += -minTargetDistance
                # elif len(self.currentCapsule) == 0 and (len(currentSafeExitA + currentSafeCapsuleA) == 0 and disA < 3) or (len(currentSafeExitB + currentSafeCapsuleB) == 0 and disB < 3):
                #     minEnemyDistance = min([self.getMazeDistance(self.nextPos, enemy.getPosition()) for enemy in self.enemies])
                #     score += 10 * minEnemyDistance
                #     if self.dead():
                #         score += 20000
                else:
                    targets = self.homeBoundary
                    maxTargetDistance = max([self.getMazeDistance(self.nextPos, target) for target in targets])
                    score += -2 * maxTargetDistance
                    minEnemyDistance = min([self.getMazeDistance(self.nextPos, enemy.getPosition()) for enemy in self.enemies])
                    score += minEnemyDistance
                    score += -2 * self.hutongMap.value(self.nextPos)
            if len(currentSafeList) > 0 and len(nextSafeList) == 0:
                score += -2000
            if len(currentSafeList) == 0 and len(nextSafeList) > 0:
                score += 2000
            if len(currentSafeList) > 0 and len(nextSafeList) > 0:
                homeDis = [(self.getMazeDistance(self.currentPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                capsuleDis = [(self.getMazeDistance(self.currentPos, capsule), capsule) for capsule in currentSafeList if not capsule in self.enemyBoundary]
                if len(homeDis) == 0:
                    backDisList = []
                    for capsule in capsuleDis:
                        dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                        backDisList.append(dis)
                    backDis = min(backDisList)
                else:
                    backDis = min(homeDis)[0]
                if gameState.data.timeleft < 4 * (backDis + 4):
                    homeDisNext = [(self.getMazeDistance(self.nextPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                    capsuleDisNext = [(self.getMazeDistance(self.nextPos, capsule), capsule) for capsule in currentSafeList if not capsule in self.enemyBoundary]
                    if len(homeDisNext) == 0:
                        backDisListNext = []
                        for capsule in capsuleDisNext:
                            dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                            backDisListNext.append(dis)
                        backDisNext = min(backDisListNext)
                    else:
                        backDisNext = min(homeDisNext)[0]
                    score += -backDisNext
                else:
                    # score += 100 * (len(self.currentFood) - len(self.nextFood))
                    if disA > 5 and disB > 5:
                        disTeamA = max([self.getMazeDistance(self.teamMatePos, self.enemies[0].getPosition()), self.enemies[0].scaredTimer])
                        disTeamB = max([self.getMazeDistance(self.teamMatePos, self.enemies[1].getPosition()), self.enemies[1].scaredTimer])
                        currentSafeExitTeamA = self.safe(self.teamMateState, self.enemies[0])
                        currentSafeCapsuleTeamA = self.safe(self.teamMateState, self.enemies[0], self.currentCapsule)
                        currentSafeExitTeamB = self.safe(self.teamMateState, self.enemies[1])
                        currentSafeCapsuleTeamB = self.safe(self.teamMateState, self.enemies[1], self.currentCapsule)
                        # safeCapsule = [capsule for capsule in self.currentCapsule if capsule in nextSafeList]
                        if self.teamMateState.isPacman and len(self.currentCapsule) > 0 and ((disTeamA < 5 and len(currentSafeExitTeamA + currentSafeCapsuleTeamA) == 0) or (disTeamB < 5 and len(currentSafeExitTeamB + currentSafeCapsuleTeamB) == 0)):
                            minCapsuleDistance = min([self.getMazeDistance(self.nextPos, capsule) for capsule in self.currentCapsule])
                            score += -minCapsuleDistance
                        elif len(self.currentFood) > 2:
                            if first:
                                foodFarFromEnemies = [self.getMazeDistance(self.nextPos, food) for food in self.currentFood if (self.getMazeDistance(food, self.enemies[0].getPosition()) > 5 or self.enemies[0].scaredTimer > 5) and (self.getMazeDistance(food, self.enemies[1].getPosition()) > 5 or self.enemies[1].scaredTimer > 5)]
                                if len(foodFarFromEnemies) > 0:
                                    minFoodDistance = min(foodFarFromEnemies)
                                else:
                                    minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in self.currentFood])
                                score += - minFoodDistance
                            else:
                                myFood = [food for food in self.currentFood if self.getMazeDistance(self.currentPos, food) < self.getMazeDistance(self.teamMatePos, food)]
                                if len(myFood) == 0:
                                    teamFood = self.greedyFoodList(self.teamMatePos, self.currentFood)
                                    myFood = teamFood[int(-len(teamFood)/2):-1]
                                if len(myFood) > 0 and len(self.currentFood) >= 6:
                                    foodFarFromEnemies = [self.getMazeDistance(self.nextPos, food) for food in myFood if (self.getMazeDistance(food, self.enemies[0].getPosition()) > 5 or self.enemies[0].scaredTimer > 5) and (self.getMazeDistance(food, self.enemies[1].getPosition()) > 5 or self.enemies[1].scaredTimer > 5)]
                                    if len(foodFarFromEnemies) > 0:
                                        minFoodDistance = min(foodFarFromEnemies)
                                    else:
                                        minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in myFood])
                                    score += - minFoodDistance
                                else:
                                    minHomeDis = min([self.getMazeDistance(self.nextPos, home) for home in currentSafeList])
                                    score += - minHomeDis
                        else:
                            homeDisNext = [(self.getMazeDistance(self.nextPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                            capsuleDisNext = [(self.getMazeDistance(self.nextPos, capsule), capsule) for capsule in currentSafeList if not capsule in self.enemyBoundary]
                            if len(homeDisNext) == 0:
                                backDisListNext = []
                                for capsule in capsuleDisNext:
                                    dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                                    backDisListNext.append(dis)
                                backDisNext = min(backDisListNext)
                            else:
                                backDisNext = min(homeDisNext)[0]
                            score += -backDisNext
                    else:
                        disTeamA = max([self.getMazeDistance(self.teamMatePos, self.enemies[0].getPosition()), self.enemies[0].scaredTimer])
                        disTeamB = max([self.getMazeDistance(self.teamMatePos, self.enemies[1].getPosition()), self.enemies[1].scaredTimer])
                        currentSafeExitTeamA = self.safe(self.teamMateState, self.enemies[0])
                        currentSafeCapsuleTeamA = self.safe(self.teamMateState, self.enemies[0], self.currentCapsule)
                        currentSafeExitTeamB = self.safe(self.teamMateState, self.enemies[1])
                        currentSafeCapsuleTeamB = self.safe(self.teamMateState, self.enemies[1], self.currentCapsule)

                        minDis = 999
                        currentSafeFood = [food for food in self.currentFood if self.hutongMap.value(food) == 0]
                        if len(currentSafeFood) > 2:
                            minSafeFoodDistance = min([self.getMazeDistance(self.nextPos, safeFood) for safeFood in currentSafeFood])
                            minDis = minSafeFoodDistance
                        safeCapsule = [capsule for capsule in self.currentCapsule if capsule in nextSafeList]
                        if len(safeCapsule) > 0:
                            minCapsuleDistance = min([self.getMazeDistance(self.nextPos, capsule) for capsule in safeCapsule])
                            if minCapsuleDistance < minDis:
                                minDis = minCapsuleDistance
                            elif self.teamMateState.isPacman and len(self.currentCapsule) > 0 and ((disTeamA < 5 and len(currentSafeExitTeamA + currentSafeCapsuleTeamA) == 0) or (disTeamB < 5 and len(currentSafeExitTeamB + currentSafeCapsuleTeamB) == 0)):
                                minDis = minCapsuleDistance
                        if minDis == 999:
                            homeDisNext = [(self.getMazeDistance(self.nextPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                            capsuleDisNext = [(self.getMazeDistance(self.nextPos, capsule), capsule) for capsule in currentSafeList if not capsule in self.enemyBoundary]
                            if len(homeDisNext) == 0:
                                backDisListNext = []
                                for capsule in capsuleDisNext:
                                    dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                                    backDisListNext.append(dis)
                                backDisNext = min(backDisListNext)
                            else:
                                backDisNext = min(homeDisNext)[0]
                            score += -backDisNext
                        else:
                            score += -minDis
        if action == Directions.STOP:
            score += -100
        return score

    def defence(self, gameState, action, invader = None):
        score = 0
        if self.dead():
            score += -10000
        disA = self.getMazeDistance(self.currentPos, self.enemies[0].getPosition())
        disB = self.getMazeDistance(self.currentPos, self.enemies[1].getPosition())
        if len(self.invaders) == 1:
            target = self.invaders[0]
        else:
            if disA < disB:
                target = self.enemies[0]
            else:
                target = self.enemies[1]
        score += -self.getMazeDistance(self.nextPos, target.getPosition())
        currentSafeExitByTeam = self.safe(target, self.teamMateState, first = False, strict = False)
        currentSafeCapsuleByTeam = self.safe(target, self.teamMateState, self.myCapsule, first = False, strict = False)
        if target.isPacman and not self.teamMateState.isPacman and len(currentSafeExitByTeam + currentSafeCapsuleByTeam) == 0 and self.getMazeDistance(self.teamMatePos, target.getPosition()) <= 5:
            if target == self.enemies[0]:
                target = self.enemies[1]
            elif target == self.enemies[1]:
                target = self.enemies[0]
        elif self.getMazeDistance(self.teamMatePos, target.getPosition()) < self.getMazeDistance(self.currentPos, target.getPosition()):
            if target == self.enemies[0]:
                target = self.enemies[1]
            elif target == self.enemies[1]:
                target = self.enemies[0]
        if target.isPacman:
            currentSafeExit = self.safe(target, self.currentState, first = False, strict = False)
            currentSafeCapsule = self.safe(target, self.currentState, self.myCapsule, first = False, strict = False)
            nextSafeExit = self.safe(target, self.nextState, strict = False)
            nextSafeCapsule = self.safe(target, self.nextState, self.myCapsule, strict = False)
            if len(nextSafeExit) == 0:
                score += 100
            if len(nextSafeCapsule) == 0:
                score += 200
            nearExit = min([(self.getMazeDistance(target.getPosition(), exit), exit) for exit in self.homeBoundary])[1]
            if not nearExit in nextSafeExit:
                score += 10
            score += -0.5 * self.getMazeDistance(self.nextPos, nearExit)
            if self.currentState.scaredTimer == 0 and len(currentSafeExit + currentSafeCapsule) > 0 and self.nextPos == target.getPosition():
                score += 500
        return score

    def greedyFoodList(self, pos, foodList):
        if len(foodList) == 0:
            return []
        nextFood = min([(self.getMazeDistance(pos, food), food) for food in foodList])[1]
        foodList.remove(nextFood)
        return [nextFood] + self.greedyFoodList(nextFood, foodList)

    def dead(self):
        if self.nextPos == self.start:
            return True
        oneDisEnemy = [i for i in self.enemies if self.getMazeDistance(self.nextPos, i.getPosition()) <= 1]
        if len(oneDisEnemy) > 0 and self.nextState.scaredTimer > 0 and not self.nextState.isPacman:
            return True
        oneDisEnemyNotScared = [i for i in oneDisEnemy if i.scaredTimer <= 1]
        if len(oneDisEnemyNotScared) > 0 and self.nextState.isPacman:
            return True
        return False

    def safe(self, a, b, targets = None, first = True, strict = True, food = False):
        safeTargets = []
        if targets == None:
            if a.getPosition()[0] <= self.redW:
                targets = self.leftBoundary
            else:
                targets = self.rightBoundary
        for target in targets:
            aDis = self.getMazeDistance(a.getPosition(), target)
            bDis = self.getMazeDistance(b.getPosition(), target)
            fast = 0
            if not first:
                fast += 1
            if strict:
                fast += 1
            if food:
                fast += self.hutongMap.value(target)
            if aDis <= bDis - fast or aDis <= b.scaredTimer - fast:
                safeTargets.append(target)
        return safeTargets

class HeuristicAgentAdvance(TestHeuristicAgent):
    def evaluate(self, gameState, action):
        self.currentState = gameState.getAgentState(self.index)
        self.currentPos = self.currentState.getPosition()
        self.successor = self.beliefGameState.generateSuccessor(self.index, action)
        self.nextState = self.successor.getAgentState(self.index)
        self.nextPos = self.nextState.getPosition()
        self.currentFood = self.getFood(gameState).asList()
        self.nextFood = self.getFood(self.successor).asList()
        self.currentCapsule = self.getCapsules(gameState)
        self.myCapsule = self.getCapsulesYouAreDefending(gameState)
        self.myFood = self.getFoodYouAreDefending(gameState).asList()

        self.teamMateState = gameState.getAgentState(self.teamMate)
        self.teamMatePos = self.teamMateState.getPosition()

        self.enemies = [self.beliefGameState.getAgentState(i) for i in self.getOpponents(gameState)]
        if self.red:
            self.invaders = [i for i in self.enemies if i.getPosition()[0] <= self.redW]
            self.defenders = [i for i in self.enemies if i.getPosition()[0] >= self.blueW]
        else:
            self.invaders = [i for i in self.enemies if i.getPosition()[0] >= self.blueW]
            self.defenders = [i for i in self.enemies if i.getPosition()[0] <= self.redW]

        safeCapsuleByMeA = self.safe(self.enemies[0], self.currentState, self.myCapsule, first = False, strict = False)
        safeCapsuleByMeB = self.safe(self.enemies[1], self.currentState, self.myCapsule, first = False, strict = False)
        safeCapsuleByTeamA = self.safe(self.enemies[0], self.teamMateState, self.myCapsule, strict = False)
        safeCapsuleByTeamB = self.safe(self.enemies[1], self.teamMateState, self.myCapsule, strict = False)
        safeFoodByMeA = self.safe(self.enemies[0], self.currentState, self.myFood, first = False, strict = False, food = True)
        safeFoodByMeB = self.safe(self.enemies[1], self.currentState, self.myFood, first = False, strict = False, food = True)
        safeFoodByTeamA = self.safe(self.enemies[0], self.currentState, self.myFood, first = False, strict = False, food = True)
        safeFoodByTeamB = self.safe(self.enemies[1], self.currentState, self.myFood, first = False, strict = False, food = True)
        if self.currentState.isPacman and abs(self.currentPos[0] - self.start[0]) > self.w + 3:
            return self.offence(gameState, action)
        # if len(self.myCapsule) > 0 and safeCapsuleByMeA:

        if len(self.invaders) == 0:
            if len(self.currentFood) == 0:
                return self.offence(gameState, action)

            minFoodDistanceMe = min([self.getMazeDistance(self.currentPos, food) for food in self.currentFood])
            minFoodDistanceTeam = min([self.getMazeDistance(self.teamMatePos, food) for food in self.currentFood])
            if minFoodDistanceMe <= minFoodDistanceTeam:
                return self.offence(gameState, action)
            else:
                return self.defence(gameState, action)
        elif len(self.invaders) >= 1:
            myClosedEnemy = []
            teamClosedEnemy = []
            for invader in self.invaders:
                safeCapsuleByMe = self.safe(invader, self.currentState, self.myCapsule, first = False, strict = False)
                safeExitByMe = self.safe(invader, self.currentState, first = False, strict = False)
                safeCapsuleByTeam = self.safe(invader, self.teamMateState, self.myCapsule, strict = False)
                safeExitByTeam = self.safe(invader, self.teamMateState, strict = False)
                if len(safeCapsuleByMe) == 0 and len(safeExitByMe) == 0:
                    myClosedEnemy.append(invader)
                if len(safeCapsuleByTeam) == 0 and len(safeExitByTeam) == 0:
                    teamClosedEnemy.append(invader)
            if len(myClosedEnemy) > 0 and len(teamClosedEnemy) > 0:
                if myClosedEnemy == teamClosedEnemy or self.getScore(gameState) < 0:
                    if len(self.currentFood) == 0:
                        return self.offence(gameState, action)
                    minFoodDistanceMe = min([self.getMazeDistance(self.currentPos, food) for food in self.currentFood])
                    minFoodDistanceTeam = min([self.getMazeDistance(self.teamMatePos, food) for food in self.currentFood])
                    if minFoodDistanceMe <= minFoodDistanceTeam:
                        return self.offence(gameState, action)
                    else:
                        return self.defence(gameState, action, invader)
                elif len(myClosedEnemy) == 1:
                    return self.defence(gameState, action, myClosedEnemy[0])
                elif len(teamClosedEnemy) == 1:
                    target = [a for a in myClosedEnemy if not a in teamClosedEnemy][0]
                    return self.defence(gameState, action, target)
            elif len(myClosedEnemy) == 0 and len(teamClosedEnemy) > 0:
                return self.offence(gameState, action)
            elif len(myClosedEnemy) > 0 and len(teamClosedEnemy) == 0:
                return self.defence(gameState, action, invader)
            elif len(myClosedEnemy) == 0 and len(teamClosedEnemy) == 0:
                if len(self.currentFood) == 0:
                    return self.offence(gameState, action)
                minFoodDistanceMe = min([self.getMazeDistance(self.currentPos, food) for food in self.currentFood])
                minFoodDistanceTeam = min([self.getMazeDistance(self.teamMatePos, food) for food in self.currentFood])
                if minFoodDistanceMe <= minFoodDistanceTeam:
                    return self.offence(gameState, action)
                else:
                    return self.defence(gameState, action)

    def offence(self, gameState, action):
        score = 0
        if self.dead():
            score += -10000
        if not self.currentState.isPacman:
            # 加上快的胶囊
            minFoodDistance = 999
            fasterFoodA = self.safe(self.currentState, self.enemies[0], self.currentFood, first = False, food = True)
            fasterFoodB = self.safe(self.currentState, self.enemies[1], self.currentFood, first = False, food = True)
            fasterFood = [food for food in fasterFoodA if food in fasterFoodB]
            if len(fasterFood) > 0:
                minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in fasterFood])
            elif len(fasterFoodA) > len(fasterFoodB) and len(fasterFoodB) > 0:
                minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in fasterFoodB])
            elif len(fasterFoodA) < len(fasterFoodB) and len(fasterFoodA) > 0:
                minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in fasterFoodA])
            else:
                # ？
                return self.defence(gameState, action)
            if not minFoodDistance == 999:
                score += -minFoodDistance
        elif self.currentPos in self.enemyBoundary and self.currentState.numCarrying > 0 and self.nextPos in self.homeBoundary:
            score += 5000
        else:
            currentSafeExitA = self.safe(self.currentState, self.enemies[0])
            currentSafeCapsuleA = self.safe(self.currentState, self.enemies[0], self.currentCapsule)
            currentSafeExitB = self.safe(self.currentState, self.enemies[1])
            currentSafeCapsuleB = self.safe(self.currentState, self.enemies[1], self.currentCapsule)
            nextSafeExitA = self.safe(self.nextState, self.enemies[0], first = False)
            nextSafeCapsuleA = self.safe(self.nextState, self.enemies[0], self.currentCapsule, first = False)
            nextSafeExitB = self.safe(self.nextState, self.enemies[1], first = False)
            nextSafeCapsuleB = self.safe(self.nextState, self.enemies[1], self.currentCapsule, first = False)

            disA = max([self.getMazeDistance(self.currentPos, self.enemies[0].getPosition()), self.enemies[0].scaredTimer])
            disB = max([self.getMazeDistance(self.currentPos, self.enemies[1].getPosition()), self.enemies[1].scaredTimer])
            if disA >= 15 and disB >= 15:
                currentSafeList = self.enemyBoundary
                nextSafeList = self.enemyBoundary
            if disA < 15 and disB >= 15:
                currentSafeList = currentSafeExitA + currentSafeCapsuleA
                nextSafeList = nextSafeExitA + nextSafeCapsuleA
            if disA >= 15 and disB < 15:
                currentSafeList = currentSafeExitB + currentSafeCapsuleB
                nextSafeList = nextSafeExitB + nextSafeCapsuleB
            if disA < 15 and disB < 15:
                currentSafeList = [p for p in (currentSafeExitA + currentSafeCapsuleA) if p in (currentSafeExitB + currentSafeCapsuleB)]
                nextSafeList = [p for p in (nextSafeExitA + nextSafeCapsuleA) if p in (nextSafeExitB + nextSafeCapsuleB)]
            if len(currentSafeList) == 0 and len(nextSafeList) == 0:
                if len(currentSafeExitA + currentSafeCapsuleA) < len(currentSafeExitB + currentSafeCapsuleB):
                    targets = (currentSafeExitA + currentSafeCapsuleA)
                else:
                    targets = (currentSafeExitB + currentSafeCapsuleB)
                if len(targets) > 0:
                    minTargetDistance = min([self.getMazeDistance(self.nextPos, target) for target in targets])
                    score += -minTargetDistance
                elif disA > 1 and disB > 1 :
                    minEnemyDistance = min([self.getMazeDistance(self.nextPos, enemy.getPosition()) for enemy in self.enemies])
                    score += -minEnemyDistance
                    score += -2 * self.hutongMap.value(self.nextPos)
                else:
                    minEnemyDistance = min([self.getMazeDistance(self.nextPos, enemy.getPosition()) for enemy in self.enemies])
                    score += 10 * minEnemyDistance
                    if self.dead():
                        score += 20000
            if len(currentSafeList) > 0 and len(nextSafeList) == 0:
                score += -1000
            if len(currentSafeList) == 0 and len(nextSafeList) > 0:
                score += 1000
            if len(currentSafeList) > 0 and len(nextSafeList) > 0:
                homeDis = [(self.getMazeDistance(self.currentPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                capsuleDis = [(self.getMazeDistance(self.currentPos, capsule), capsule) for capsule in currentSafeList if capsule in self.currentCapsule]
                if len(homeDis) == 0:
                    backDisList = []
                    for capsule in capsuleDis:
                        dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                        backDisList.append(dis)
                    backDis = min(backDisList)
                else:
                    backDis = min(homeDis)[0]
                if gameState.data.timeleft < 4 * (backDis + 4):
                    homeDisNext = [(self.getMazeDistance(self.nextPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                    capsuleDisNext = [(self.getMazeDistance(self.nextPos, capsule), capsule) for capsule in currentSafeList if capsule in self.currentCapsule]
                    if len(homeDisNext) == 0:
                        backDisListNext = []
                        for capsule in capsuleDisNext:
                            dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                            backDisListNext.append(dis)
                        backDisNext = min(backDisListNext)
                    else:
                        backDisNext = min(homeDisNext)[0]
                    score += -backDisNext
                else:
                    score += 100 * (len(self.currentFood) - len(self.nextFood))
                    if disA > 5 and disB > 5:
                        if len(self.currentFood) > 2:
                            minFoodDistance = min([self.getMazeDistance(self.nextPos, food) for food in self.currentFood])
                            score += - minFoodDistance
                        else:
                            minHomeDis = min([self.getMazeDistance(self.nextPos, home) for home in currentSafeList])
                            score += - minHomeDis
                    else:
                        minDis = 999
                        currentSafeFood = [food for food in self.currentFood if self.hutongMap.value(food) == 0]
                        if len(currentSafeFood) > 0:
                            minSafeFoodDistance = min([self.getMazeDistance(self.nextPos, safeFood) for safeFood in currentSafeFood])
                            minDis = minSafeFoodDistance
                        safeCapsule = [capsule for capsule in self.currentCapsule if capsule in nextSafeList]
                        if len(safeCapsule) > 0:
                            minCapsuleDistance = min([self.getMazeDistance(self.nextPos, capsule) for capsule in safeCapsule])
                            if minCapsuleDistance < minDis:
                                minDis = minCapsuleDistance
                        if minDis == 999:
                            homeDisNext = [(self.getMazeDistance(self.nextPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                            capsuleDisNext = [(self.getMazeDistance(self.nextPos, capsule), capsule) for capsule in currentSafeList if capsule in self.currentCapsule]
                            if len(homeDisNext) == 0:
                                backDisListNext = []
                                for capsule in capsuleDisNext:
                                    dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                                    backDisListNext.append(dis)
                                backDisNext = min(backDisListNext)
                            else:
                                backDisNext = min(homeDisNext)[0]
                            score += -backDisNext
                        else:
                            score += -minDis
        if action == Directions.STOP:
            score += -100
        return score

    def defence(self, gameState, action, invader = None):
        score = 0
        if self.dead():
            score += -10000
        if self.currentState.isPacman:
            currentSafeExitA = self.safe(self.currentState, self.enemies[0])
            currentSafeCapsuleA = self.safe(self.currentState, self.enemies[0], self.currentCapsule)
            currentSafeExitB = self.safe(self.currentState, self.enemies[1])
            currentSafeCapsuleB = self.safe(self.currentState, self.enemies[1], self.currentCapsule)
            nextSafeExitA = self.safe(self.nextState, self.enemies[0], first = False)
            nextSafeCapsuleA = self.safe(self.nextState, self.enemies[0], self.currentCapsule, first = False)
            nextSafeExitB = self.safe(self.nextState, self.enemies[1], first = False)
            nextSafeCapsuleB = self.safe(self.nextState, self.enemies[1], self.currentCapsule, first = False)
            currentSafeList = [p for p in (currentSafeExitA + currentSafeCapsuleA) if p in (currentSafeExitB + currentSafeCapsuleB)]
            nextSafeList = [p for p in (nextSafeExitA + nextSafeCapsuleA) if p in (nextSafeExitB + nextSafeCapsuleB)]
            if len(currentSafeList) == 0 and len(nextSafeList) == 0:
                if len(currentSafeExitA + currentSafeCapsuleA) > len(currentSafeExitB + currentSafeCapsuleB):
                    targets = (currentSafeExitA + currentSafeCapsuleA)
                else:
                    targets = (currentSafeExitB + currentSafeCapsuleB)
                if len(targets) > 0:
                    minTargetDistance = min([self.getMazeDistance(self.nextPos, target) for target in targets])
                    score += -minTargetDistance
                else:
                    minEnemyDistance = min([self.getMazeDistance(self.nextPos, enemy.getPosition()) for enemy in self.enemies])
                    score += -minEnemyDistance
                    score += -2 * self.hutongMap.value(self.nextPos)
                    if self.hutongMap.isOnExit(self.enemies[0].getPosition(), self.currentPos) or self.hutongMap.isOnExit(self.enemies[1].getPosition(), self.currentPos):
                        score += 10 * minEnemyDistance
                        if self.dead():
                            score += 20000
            if len(currentSafeList) > 0 and len(nextSafeList) == 0:
                score += -1000
            if len(currentSafeList) == 0 and len(nextSafeList) > 0:
                score += 1000
            if len(currentSafeList) > 0 and len(nextSafeList) > 0:
                if self.red:
                    if self.enemies[0].getPosition()[0] <= self.currentPos[0] and self.enemies[1].getPosition()[0] <= self.currentPos[0] or self.currentState.numCarrying >= 3:
                        homeDis = [(self.getMazeDistance(self.currentPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                        capsuleDis = [(self.getMazeDistance(self.currentPos, capsule), capsule) for capsule in currentSafeList if capsule in self.currentCapsule]
                        if len(homeDis) == 0:
                            backDisList = []
                            for capsule in capsuleDis:
                                dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                                backDisList.append(dis)
                            backDis = min(backDisList)
                        else:
                            backDis = min(homeDis)[0]
                        homeDisNext = [(self.getMazeDistance(self.nextPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                        capsuleDisNext = [(self.getMazeDistance(self.nextPos, capsule), capsule) for capsule in currentSafeList if capsule in self.currentCapsule]
                        if len(homeDisNext) == 0:
                            backDisListNext = []
                            for capsule in capsuleDisNext:
                                dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                                backDisListNext.append(dis)
                            backDisNext = min(backDisListNext)
                        else:
                            backDisNext = min(homeDisNext)[0]
                        score += 5 * (backDis - backDisNext)
                else:
                    if self.enemies[0].getPosition()[0] >= self.currentPos[0] and self.enemies[1].getPosition()[0] >= self.currentPos[0] or self.currentState.numCarrying >= 3:
                        homeDis = [(self.getMazeDistance(self.currentPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                        capsuleDis = [(self.getMazeDistance(self.currentPos, capsule), capsule) for capsule in currentSafeList if capsule in self.currentCapsule]
                        if len(homeDis) == 0:
                            backDisList = []
                            for capsule in capsuleDis:
                                dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                                backDisList.append(dis)
                            backDis = min(backDisList)
                        else:
                            backDis = min(homeDis)[0]
                        homeDisNext = [(self.getMazeDistance(self.nextPos, home), home) for home in currentSafeList if home in self.enemyBoundary]
                        capsuleDisNext = [(self.getMazeDistance(self.nextPos, capsule), capsule) for capsule in currentSafeList if capsule in self.currentCapsule]
                        if len(homeDisNext) == 0:
                            backDisListNext = []
                            for capsule in capsuleDisNext:
                                dis = capsule[0] + min([self.getMazeDistance(capsule[1], home) for home in self.homeBoundary])
                                backDisListNext.append(dis)
                            backDisNext = min(backDisListNext)
                        else:
                            backDisNext = min(homeDisNext)[0]
                        score += 5 * (backDis - backDisNext)
            if self.currentPos in self.enemyBoundary and self.currentState.numCarrying > 0 and self.nextPos in self.homeBoundary:
                score += 20
        if not invader == None:
            currentSafeExit = self.safe(invader, self.currentState, first = False, strict = False)
            currentSafeCapsule = self.safe(invader, self.currentState, self.myCapsule, first = False, strict = False)
            nextSafeExit = self.safe(invader, self.nextState, strict = False)
            nextSafeCapsule = self.safe(invader, self.nextState, self.myCapsule, strict = False)
            if len(nextSafeExit) == 0:
                score += 100
            if len(nextSafeCapsule) == 0:
                score += 200
        else:
            if len(self.invaders) == 0:
                currentSafeCapsuleA = self.safe(self.enemies[0], self.currentState, self.myCapsule, first = False, strict = False)
                nextSafeCapsuleA = self.safe(self.enemies[0], self.nextState, self.myCapsule, strict = False)
                currentSafeCapsuleB = self.safe(self.enemies[1], self.currentState, self.myCapsule, first = False, strict = False)
                nextSafeCapsuleB = self.safe(self.enemies[1], self.nextState, self.myCapsule, strict = False)
                if len(nextSafeCapsuleA) == 0:
                    score += 200
                if len(nextSafeCapsuleB) == 0:
                    score += 200
                score += 50 * (min([len(currentSafeCapsuleA), len(currentSafeCapsuleB)]) - min([len(nextSafeCapsuleA), len(nextSafeCapsuleB)]))
                score += min([self.getMazeDistance(self.currentPos, self.enemies[0].getPosition()), self.getMazeDistance(self.currentPos, self.enemies[1].getPosition())])\
                    - min([self.getMazeDistance(self.nextPos, self.enemies[0].getPosition()), self.getMazeDistance(self.nextPos, self.enemies[1].getPosition())])
            if len(self.invaders) == 1:
                target = self.invaders[0]
                currentSafeExit = self.safe(target, self.currentState, first = False, strict = False)
                currentSafeCapsule = self.safe(target, self.currentState, self.myCapsule, first = False, strict = False)
                nextSafeExit = self.safe(target, self.nextState, strict = False)
                nextSafeCapsule = self.safe(target, self.nextState, self.myCapsule, strict = False)
                if len(nextSafeExit) == 0:
                    score += 100
                if len(nextSafeCapsule) == 0:
                    score += 200
                safeFoodByMe = self.safe(target, self.currentState, self.myFood, first = False, strict = False, food = True)
                nextSafeFoodByMe = self.safe(target, self.nextState, self.myFood, first = False, strict = False, food = True)
                score += len(safeFoodByMe) - len(nextSafeFoodByMe)
                score += self.getMazeDistance(self.currentPos, target.getPosition()) - self.getMazeDistance(self.nextPos, target.getPosition())
            if len(self.invaders) == 2:
                currentSafeExitA = self.safe(self.invaders[0], self.currentState, first = False, strict = False)
                currentSafeCapsuleA = self.safe(self.invaders[0], self.currentState, self.myCapsule, first = False, strict = False)
                nextSafeExitA = self.safe(self.invaders[0], self.nextState, strict = False)
                nextSafeCapsuleA = self.safe(self.invaders[0], self.nextState, self.myCapsule, strict = False)
                currentSafeExitB = self.safe(self.invaders[1], self.currentState, first = False, strict = False)
                currentSafeCapsuleB = self.safe(self.invaders[1], self.currentState, self.myCapsule, first = False, strict = False)
                nextSafeExitB = self.safe(self.invaders[1], self.nextState, strict = False)
                nextSafeCapsuleB = self.safe(self.invaders[1], self.nextState, self.myCapsule, strict = False)
                if len(nextSafeExitA) == 0:
                    score += 100
                if len(nextSafeCapsuleA) == 0:
                    score += 200
                if len(nextSafeExitB) == 0:
                    score += 100
                if len(nextSafeCapsuleB) == 0:
                    score += 200
                safeFoodByMeA = self.safe(self.invaders[0], self.currentState, self.myFood, first = False, strict = False, food = True)
                safeFoodByMeB = self.safe(self.invaders[1], self.currentState, self.myFood, first = False, strict = False, food = True)
                nextSafeFoodByMeA = self.safe(self.invaders[0], self.nextState, self.myFood, first = False, strict = False, food = True)
                nextSafeFoodByMeB = self.safe(self.invaders[1], self.nextState, self.myFood, first = False, strict = False, food = True)
                score += min([len(safeFoodByMeA),len(safeFoodByMeB)]) - min([len(nextSafeFoodByMeA),len(nextSafeFoodByMeB)])
                score += min([self.getMazeDistance(self.currentPos, self.invaders[0].getPosition()), self.getMazeDistance(self.currentPos, self.invaders[1].getPosition())])\
                    - min([self.getMazeDistance(self.nextPos, self.invaders[0].getPosition()), self.getMazeDistance(self.nextPos, self.invaders[1].getPosition())])
        if len(self.currentFood) > 2 and self.currentState.numCarrying < 3:
            myFood = [food for food in self.currentFood if self.getMazeDistance(self.currentPos, food) < self.getMazeDistance(self.teamMatePos, food)]
            if len(myFood) > 0:
                minDistance = 2 * min([self.getMazeDistance(self.nextPos, food) for food in myFood])
                score += -minDistance
        if self.currentPos in self.homeBoundary or self.currentPos in self.enemyBoundary and self.getMazeDistance(self.currentPos, self.teamMatePos) <= 5:
            score += 3 * self.getMazeDistance(self.nextPos, self.teamMatePos)
        return score

    def dead(self):
        if self.nextPos == self.start:
            return True
        oneDisEnemy = [i for i in self.enemies if self.getMazeDistance(self.nextPos, i.getPosition()) <= 1]
        if len(oneDisEnemy) > 0 and self.nextState.scaredTimer > 0 and not self.nextState.isPacman:
            return True
        oneDisEnemyNotScared = [i for i in oneDisEnemy if i.scaredTimer <= 1]
        if len(oneDisEnemyNotScared) > 0 and self.nextState.isPacman:
            return True
        return False

    def safe(self, a, b, targets = None, first = True, strict = True, food = False):
        safeTargets = []
        if targets == None:
            if a.getPosition()[0] <= self.redW:
                targets = self.leftBoundary
            else:
                targets = self.rightBoundary
        for target in targets:
            aDis = self.getMazeDistance(a.getPosition(), target)
            bDis = self.getMazeDistance(b.getPosition(), target)
            fast = 0
            if not first:
                fast += 1
            if strict:
                fast += 1
            if food:
                fast += self.hutongMap.value(target)
            if aDis <= bDis - fast or aDis <= b.scaredTimer - fast:
                safeTargets.append(target)
        return safeTargets

class HeuristicAgentO(TestHeuristicAgent):
    def evaluate(self, gameState, action):
        score = 0

        currentState = gameState.getAgentState(self.index)
        currentPos = currentState.getPosition()
        successor = self.beliefGameState.generateSuccessor(self.index, action)
        nextState = successor.getAgentState(self.index)
        nextPos = nextState.getPosition()
        currentFood = self.getFood(gameState).asList()
        nextFood = self.getFood(successor).asList()

        nearestDefender = None
        minDefendersDistance = 9999
        for i in self.getOpponents(gameState):
            enemy = self.beliefGameState.getAgentState(i)
            dis = self.getMazeDistance(currentPos, enemy.getPosition())
            if enemy.scaredTimer > 0 and enemy.scaredTimer > dis:
                dis = enemy.scaredTimer
            if dis < minDefendersDistance and not enemy.isPacman:
                nearestDefender = enemy
                minDefendersDistance = dis

        if minDefendersDistance > 5:
            if len(currentFood) > 0 and currentState.numCarrying < 10:
                score += 100 * (len(currentFood) - len(nextFood))
                if len(nextFood) > 0:
                    minFoodDistance = min([self.getMazeDistance(nextPos, food) for food in nextFood])
                    score += -minFoodDistance
            else:
                score += -self.DistanceToHome.value(nextPos)[0][0]
        else:
            currentSafeFood = [food for food in currentFood if self.hutongMap.value(food) == 0]
            nextSafeFood = [food for food in nextFood if self.hutongMap.value(food) == 0]
            currentCapsule = self.getCapsules(gameState)
            nextCapsule = self.getCapsules(successor)
            if len(currentSafeFood) > 0 and currentState.numCarrying < 5:
                score += 100 * (len(currentSafeFood) - len(nextSafeFood))
                score += 200 * (len(currentCapsule) - len(nextCapsule))
                if len(nextSafeFood) > 0:
                    minTargetDistance = min([self.getMazeDistance(nextPos, safeFood) for safeFood in nextSafeFood])
                    if len(currentCapsule) > 0:
                        minCapsuleDistance = min([self.getMazeDistance(nextPos, capsule) for capsule in currentCapsule])
                        if minCapsuleDistance < minTargetDistance:
                            minTargetDistance = minCapsuleDistance
                    score += -minTargetDistance

                if self.getMazeDistance(nextPos, nearestDefender.getPosition()) == 1 or nextPos == self.start and nearestDefender.scaredTimer <= 0:
                    score += -1000
            else:
                score += -self.DistanceToHome.value(nextPos)[0][0]
                score += -2 * self.hutongMap.value(nextPos)
                if nearestDefender.scaredTimer <= 0:
                    score += 2 * self.getMazeDistance(nextPos, nearestDefender.getPosition())
                    if self.getMazeDistance(nextPos, nearestDefender.getPosition()) == 1 or nextPos == self.start:
                        score += -1000

        if action == Directions.STOP:
            score += -0.1

        return score

class HeuristicAgentD(TestHeuristicAgent):
    def evaluate(self, gameState, action):
        score = 0
        currentState = gameState.getAgentState(self.index)
        currentPos = currentState.getPosition()
        successor = self.beliefGameState.generateSuccessor(self.index, action)
        nextState = successor.getAgentState(self.index)
        nextPos = nextState.getPosition()
        currentFood = self.getFood(gameState).asList()
        nextFood = self.getFood(successor).asList()

        if currentState.scaredTimer > 10:
            return HeuristicAgentO.evaluate(self, gameState, action)

        enemies = self.getOpponents(successor)
        if self.red:
            invaders = [i for i in enemies if self.beliefGameState.getAgentPosition(i)[0] <= self.redW]
        else:
            invaders = [i for i in enemies if self.beliefGameState.getAgentPosition(i)[0] >= self.blueW]

        nearestEnemyDis, nearestEnemy = \
            min([(self.getMazeDistance(currentPos, self.beliefGameState.getAgentPosition(i)), i) for i in enemies])
        nearestEnemyPos = self.beliefGameState.getAgentPosition(nearestEnemy)
        if len(invaders) == 1:
            nearestInvader = invaders[0]
            nearestInvaderDis = self.getMazeDistance(currentPos, self.beliefGameState.getAgentPosition(nearestInvader))
            nearestInvaderPos = self.beliefGameState.getAgentPosition(nearestInvader)
        elif len(invaders) > 1:
            nearestInvaderDis, nearestInvader = \
                min([(self.getMazeDistance(currentPos, self.beliefGameState.getAgentPosition(i)), i) for i in invaders])
            nearestInvaderPos = self.beliefGameState.getAgentPosition(nearestInvader)

        if len(invaders) == 0:
            score += -self.getMazeDistance(nextPos, nearestEnemyPos)
            if nextState.isPacman:
                if self.DistanceToHome.value(nextPos)[0][0] > 1:
                    score += -self.DistanceToHome.value(nextPos)[0][0]
                if self.getMazeDistance(nextPos, nearestEnemyPos) == 1 or nextPos == self.start:
                    score += -1000
        else:
            if nextState.isPacman:
                if self.DistanceToHome.value(nextPos)[0][0] > 1:
                    score += -0.5 * self.DistanceToHome.value(nextPos)[0][0]
                if self.getMazeDistance(nextPos, nearestEnemyPos) == 1 or nextPos == self.start:
                    score += -1000
            for a in invaders:
                if self.hutongMap.isOnExit(currentPos, self.beliefGameState.getAgentPosition(a)) and action == Directions.STOP:
                    score += 1000
            if nextState.scaredTimer > 0 and self.getMazeDistance(nextPos, nearestInvaderPos) == 1:
                score += -1000
            if nextPos == self.start:
                score += -1000
            if nextPos == nearestInvaderPos:
                score += 50

            score += -self.getMazeDistance(nextPos, nearestInvaderPos)

            exitDis, exit = self.DistanceToHome.value(nearestInvaderPos)[0]
            target = exit
            if len(self.getCapsulesYouAreDefending(gameState)) > 0:
                capsuleDis, capsule = min([(self.getMazeDistance(nearestInvaderPos, capsule), capsule) for capsule in self.getCapsulesYouAreDefending(gameState)])
                target = min([(exitDis, exit), (capsuleDis, capsule)])[1]
            score += -0.1 * self.getMazeDistance(nextPos, target)

        if action == Directions.STOP:
            score += -0.01

        return score

simulateHistory = {}
recordDeep = 15
simulateNumber = 30
simulateDeep = 15
class TestHeuristicAgentOWithMCTS(TestHeuristicAgentO):
    def _chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if not defenders == []:
            values = [self.MCTSevaluate(gameState.generateSuccessor(self.index, action)) for action in actions]
        else:
            values = [self.evaluate(gameState, action) for action in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def MCTSevaluate(self, gameState):
        start = time.time()
        while time.time() -start < 0.18:
            self.MCTSsimulate(gameState, simulateDeep)
            if simulateHistory[gameState][0]>simulateNumber:
                break
        return simulateHistory[gameState][1]/simulateHistory[gameState][0]

    def MCTSsimulate(self, gameState, n):
        if gameState.getAgentState(self.index).getPosition() == self.start:
            result = self.evaluate(gameState, Directions.STOP) - 1000
            if simulateDeep - n < recordDeep:
                self.update(gameState, result)
            return result
        if n == 0:
            return self.evaluate(gameState, Directions.STOP)
        else:
            enemies = [i for i in self.getOpponents(gameState) if not gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]
            for a in self.getTeam(gameState):
                nextState = gameState.generateSuccessor(a, random.choice(gameState.getLegalActions(a)))
            for a in enemies:
                nextState = nextState.generateSuccessor(a, random.choice(nextState.getLegalActions(a)))
            result = self.MCTSsimulate(nextState, n - 1)
            if simulateDeep - n < recordDeep:
                self.update(gameState, result)
            return result

    def update(self, gameState, value):
        (x, y) = (0, 0)
        if gameState in simulateHistory:
            (x, y) = simulateHistory[gameState]
        (x, y) = (x+1, y+value)
        simulateHistory[gameState] = (x, y)

class TestHeuristicAgentD(TestHeuristicAgent):
    def evaluate(self, gameState, action):
        score = 0
        successor = gameState.generateSuccessor(self.index, action)
        myState = successor.getAgentState(self.index)
        myPos = successor.getAgentState(self.index).getPosition()

        if myState.isPacman:
            score += -200
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        score += -1000 * len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            score += -10 * min(dists)
            if successor.getAgentState(self.index).scaredTimer > 0 and min(dists) == 1:
                score += -3000
            for a in invaders:
                if self.hutongMap.isOnExit(myPos, a.getPosition()) and action == Directions.STOP:
                    score += 2000
        else:
            dists = [self.getMazeDistance(myPos, inference[a].beliefs.argMax()) for a in self.getOpponents(successor)]
            score += -10 * min(dists)
        score += -20 * self.hutongMap.value(myPos)
        if action == Directions.STOP:
            score += -100
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            score += -2
        middle = (gameState.getWalls().width/2 - 1, gameState.getWalls().height/2 - 1)
        score += -0.1 * abs(myPos[0] - middle[0])
        return score

class TestMCTSAgent(CaptureAgentWithTools):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [(self.evaluate(gameState, action), action) for action in actions]
        # print(values)
        # util.pause()
        return max(values)[1]

    def evaluate(self, gameState, action):
        result = 0
        t = 100
        d = 100
        for i in range(t):
            result += self.simulate(gameState, action, d)
        return result/t

    def simulate(self, gameState, action, n):
        nextState = gameState.generateSuccessor(self.index, action)
        nextPosition = util.nearestPoint(nextState.getAgentState(self.index).getPosition())
        if n == 0:
            return self.getScore(nextState)
        else:
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(nextState) if gameState.getAgentState(i).getPosition() != None]
            for a in enemies:
                nextState = nextState.generateSuccessor(a.index, random.choice(nextState.getLegalActions(a.index)))
            nextActions = nextState.getLegalActions(self.index)
            nextActions.remove(Directions.STOP)
            rev = Directions.REVERSE[action]
            if not gameState.hasFood(nextPosition[0], nextPosition[1]) and rev in nextActions and len(nextActions) > 1:
                nextActions.remove(rev)
            return self.simulate(nextState, random.choice(nextActions), n - 1)

class TestGameTheoryAgent(CaptureAgentWithTools):
    def _chooseAction(self, gameState):
        decisionTree = DecisionTree(self.index, self.beliefGameState, self.red)
        decisionTree.construct(10)
        return decisionTree.bestAction()

evaluateHistory = {}
class TestGameTheoryAgentWithHeuristic(CaptureAgentWithTools):
    def _chooseAction(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        enemiesIndexs = self.getOpponents(self.beliefGameState)
        enemies = [self.beliefGameState.getAgentState(i) for i in enemiesIndexs]
        distances = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies]
        minEnemiesDistance = min(distances)
        deep = 10
        if minEnemiesDistance < 10:
            near = [i for i, d in zip(enemiesIndexs, distances) if d == min(distances)][0]
            far = [i for i in enemiesIndexs if not i == near][0]
            teammate = [i for i in self.getTeam(gameState) if not i == self.index][0]
            simpleState = self.beliefGameState.deepCopy()
            simpleState.data.agentStates[far] = None
            simpleState.data.agentStates[teammate] = None

            actions = gameState.getLegalActions(self.index)
            values = [self.ab(self.beliefGameState.generateSuccessor(self.index, action), deep, -999999.9, 999999.9, False, near) for action in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return random.choice(bestActions)
        else:
            actions = gameState.getLegalActions(self.index)
            values = [self.evaluate1(gameState.generateSuccessor(self.index, action)) for action in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return random.choice(bestActions)


        # decisionTree = DecisionTree(self.index, self.beliefGameState, self.red)
        # decisionTree.construct(deep)
        # return decisionTree.bestAction(caller = self)

    def ab(self, gameState, deep, a, b, maxP, enemy):
        if deep == 0:
            return self.evaluate(gameState)
        if maxP:
            actions = gameState.getLegalActions(self.index)
            v = -999999.9
            for action in actions:
                if action == Directions.STOP:
                    continue
                value = self.ab(gameState.generateSuccessor(self.index, action), deep - 1, a, b, False, enemy)
                v = max(v, value)
                a = max(a, v)
                if b <= a:
                    break
            return v
        else:
            actions = gameState.getLegalActions(enemy)
            v = 999999.9
            for action in actions:
                if action == Directions.STOP:
                    continue
                value = self.ab(gameState.generateSuccessor(enemy, action), deep - 1, a, b, True, enemy)
                v = min(v, value)
                b = min(a, v)
                if b <= a:
                    break
            return v

    def evaluate1(self, gameState):
        score = 0

        myState = gameState.getAgentState(self.index)
        myPos = gameState.getAgentState(self.index).getPosition()

        foodList = self.getFood(gameState).asList()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        defenders = [a for a in enemies if not a.isPacman and a.scaredTimer <= 0 and a.getPosition() != None and self.getMazeDistance(myPos, a.getPosition()) <= 6]
        if len(defenders) > 0:
            score += -3000
            minDefendersDistance = min([self.getMazeDistance(myPos, defender.getPosition()) for defender in defenders])
            score += 10 * minDefendersDistance
            score += -20 * self.hutongMap.value(myPos)
            safeFood = [self.getMazeDistance(myPos, food) for food in foodList if self.hutongMap.value(food) == 0]
            score += -200 * len(safeFood)
            if not safeFood == []:
                minSafeFoodDistance = min(safeFood)
                score += -minSafeFoodDistance
            score += -0.7 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
        elif len(foodList) > 0:
            score += -100 * len(foodList)
            minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            score += -minFoodDistance
            score += -0.14 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
        score += -200 * len([a for a in enemies if a.isPacman or a.scaredTimer > 0 and a.getPosition() != None])

        return score

        # def evaluate(self, gameState):
        #     if gameState in evaluateHistory:
        #         return evaluateHistory[gameState]
        #     score = 0.0
        #     score += 1000 * gameState.getScore()
        #     score += 100 * gameState.getRedFood().count()
        #     score += -100 * gameState.getBlueFood().count()
        #     red = [gameState.getAgentState(a) for a in gameState.getRedTeamIndices()]
        #     blue = [gameState.getAgentState(a) for a in gameState.getBlueTeamIndices()]
        #     blueFood = gameState.getBlueFood().asList()
        #     redFood = gameState.getRedFood().asList()
        #     for a in red:
        #         if a.getPosition() == a.start.pos:
        #             score += -5000
        #         if not blueFood == []:
        #             redMinFoodDistance = min([self.getMazeDistance(a.getPosition(), f) for a in red for f in blueFood])
        #             score += -1 * redMinFoodDistance
        #         redMinFoodDistance = min([self.getMazeDistance(a.getPosition(), f) for a in red for f in blueFood])
        #         score += -1 * redMinFoodDistance
        #         if a.isPacman:
        #             # score += -100
        #             enemyDistance = [self.getMazeDistance(a.getPosition(), enemy.getPosition()) for enemy in blue if enemy.scaredTimer <= 0]
        #             if not enemyDistance == []:
        #                 score += 2 * min(enemyDistance)
        #         else:
        #             minEnemyDistance = min([self.getMazeDistance(a.getPosition(), enemy.getPosition()) for enemy in blue])
        #             score += -0.1 * minEnemyDistance
        #     for a in blue:
        #         if a.getPosition() == a.start.pos:
        #             score += 5000
        #         if not redFood == []:
        #             blueMinFoodDistance = min([self.getMazeDistance(a.getPosition(), f) for a in blue for f in redFood])
        #             score += 1 * blueMinFoodDistance
        #         if a.isPacman:
        #             # score += 100
        #             enemyDistance = [self.getMazeDistance(a.getPosition(), enemy.getPosition()) for enemy in red if enemy.scaredTimer <= 0]
        #             if not enemyDistance == []:
        #                 score += -2 * min(enemyDistance)
        #         else:
        #             minEnemyDistance = min([self.getMazeDistance(a.getPosition(), enemy.getPosition()) for enemy in red])
        #             score += 0.01 * minEnemyDistance
        #     evaluateHistory[gameState] = score
        #     return score

    def evaluate(self, gameState):
        score = 0
        foodList = self.getFood(gameState).asList()
        score += 1000 * self.getScore(gameState)
        teammate = [i for i in self.getTeam(gameState) if not i == self.index][0]
        teamPos = util.nearestPoint(self.beliefGameState.getAgentPosition(teammate))
        teammateFoodDis = closestFood(teamPos, self.getFood(gameState), gameState.getWalls())
        myPos = util.nearestPoint(gameState.getAgentState(self.index).getPosition())
        myFoodDis = closestFood(myPos, self.getFood(gameState), gameState.getWalls())
        if myFoodDis > teammateFoodDis:
            score += -20000

        if len(foodList) > 0:
            score += -100 * len(foodList)
            minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            score += -minFoodDistance
        return score

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

class HuTong:
    """
    This class is for a weighted map, map is a Grid. In Grid:
        True/T : is wall
        0 : is the main read
        other digits : is the deep of the HuTon.
    """
    def __init__(self, walls):
        self.huTongPath = []
        self.map = self._getHuTong(walls.deepCopy())

    def _getHuTong(self, map):
        """
        This function return a weighted huTong map, a Grid.
        """
        huTongBottomList = self._getHuTongBottom(map)
        huTongList = []
        for (x,y) in huTongBottomList:
            iniSuccessors = self._getSuccessors(x, y, map)
            singleHuTong = [(x,y)]
            (currx,curry) = iniSuccessors[0]
            curSuccessors = self._getSuccessors(currx,curry, map)
            while len (curSuccessors) == 2:
                # Each huTongBottom is from the right end of each list.
                singleHuTong.insert(0, (currx,curry))
                for next in curSuccessors:
                    if next not in singleHuTong:
                        (currx,curry) = next
                        curSuccessors = self._getSuccessors(currx,curry, map)
            for (eachx,eachy) in singleHuTong:
                map[eachx][eachy] = "e"
            # The [0] element is the cross of current huTong.
            singleHuTong.insert(0, (currx,curry))
            huTongList.append(singleHuTong)
        # get full huTongPath.
        self.huTongPath = self._getFullHuTongPath(huTongList)

        # mark map.
        for huTong in self.huTongPath:
            count = 0
            for (x,y) in huTong:
                map[x][y] = count
                count += 1
        for x in range(1, map.width-1):
            for y in range(1, map.height-1):
                if not map[x][y]:
                    map[x][y] = 0

        return map

    def _getHuTongBottom(self, map):
        """
        Return all huTongBottom position in a list, [(x,y)].
        """
        huTongBottomList = []
        for i in range(1, map.width-1):
            for j in range(1, map.height-1):
                count = len(self._getSuccessors(i, j, map))
                if count == 1 and not map[i][j]:
                    huTongBottomList.append((i,j))
        return huTongBottomList

    def _getSuccessors(self, x, y, map):
        """
        Return the successors for map[x][y] which is not wall.
        e.g. if map[x][y] = True, it is wall, else, otherwise.
        """
        successors = []
        if not map[x-1][y]: successors.append((x-1,y))
        if not map[x][y-1]: successors.append((x,y-1))
        if not map[x+1][y]: successors.append((x+1,y))
        if not map[x][y+1]: successors.append((x,y+1))
        return successors

    def _isBranchCross(self, currentHuTong, huTongList):
        """
        Return mainHuTong, huTong[0] is a branchCross.
        Return [], huTong[0] is a mainRoadCross.
        """
        for each in huTongList:
            if currentHuTong[0] in each[1:]:
                return each
        return None

    def _getFullHuTongPath(self, huTongList):
        """
        Return huTongPath from the real head without in another's body.
        """
        huTongPath = []
        for huTong in huTongList:
            while True:
                mainHuTong = self._isBranchCross(huTong, huTongList)
                if mainHuTong is None:
                    break
                else:
                    index = mainHuTong.index(huTong[0])
                    huTong = mainHuTong[:index]+huTong
            huTongPath.append(huTong)
        return huTongPath

    def value(self, pos):
        (x, y) = util.nearestPoint(pos)
        return self.map[x][y]

    def isOnExit(self, ownState, enemyState):
        """
        Return if ownState is on the exit side of enemyState.
        Each state shoud be a (x,y) position.
        """
        for path in self.huTongPath:
            if ownState in path and enemyState in path:
                if path.index(ownState) < path.index(enemyState):
                    return True
                else:
                    return False
        return False

# example:
# DistanceToHome.distanceMapping = {(1, 1): [(50, (17, 5))], (1, 2): [(49, (17, 5))]}
# init instance:
#     self.distanceToHome = DistanceToHome(gameState.getWalls(), self.getMazeDistance)
class DistanceToHome:
    """
    This class calculates the distance from each cell to its homes side.
    Return (the nearest home position, distance to that position)
    """
    def __init__(self, walls, getMazeDistance):
        self._walls = walls.deepCopy()
        self._getMazeDistance = getMazeDistance
        self.w = self._walls.width // 2 - 1
        self.h = self._walls.height
        self.distanceMapping = self._getDistance()

    def _getDistance(self):
        distanceMapping = {}
        leftBoundary, rightBoundary = self._getBoundary()
        # left side cells.
        for i in range(self._walls.width // 2):
            for j in range(self._walls.height):
                if not self._walls[i][j]:
                    distanceMapping[(i, j)] = self._getNearestHomePosition((i, j), rightBoundary)
        #  right side cells.
        for i in range(self._walls.width // 2,  self._walls.width):
            for j in range(self._walls.height):
                if not self._walls[i][j]:
                    distanceMapping[(i, j)] = self._getNearestHomePosition((i, j), leftBoundary)
        return distanceMapping

    def _getBoundary(self):
        leftBoundary = [(self.w, h) for h in range(self.h) if not self._walls[self.w][h]]
        rightBoundary = [(self.w+1, h) for h in range(self.h) if not self._walls[self.w+1][h]]
        return leftBoundary, rightBoundary

    def _getNearestHomePosition(self, position, rightBoundary):
        distance = [(self._getMazeDistance(position, b), b) for b in rightBoundary]
        distance = sorted(distance, key=lambda x:x[0])
        minDistance = distance[0][0]
        return [(d, p) for (d, p) in distance if d == minDistance]

    def value(self, pos):
        (x, y) = util.nearestPoint(pos)
        return self.distanceMapping[(x, y)]

class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        """
        total = self.total()
        if total == 0:
            return
        all = list(self.items())
        for item in all:
            self[item[0]] = item[1]/total

    def sample(self):
        value = random.random() *self.total()
        all = list(self.items())
        for item in all:
            if value > item[1]:
                value -= item[1]
            else:
                return item[0]

class InferenceModule:
    """
    An inference module tracks a belief distribution over an enemy's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, index, gameState):
        """
        Set the enemy agent for later access.
        """
        self.index = index
        self.legalPositions = gameState.getWalls().asList(False)
        self.initialAgentPosition = gameState.getInitialAgentPosition(self.index)

    def getPositionDistribution(self, gameState, pos):
        """
        Return a distribution over successor positions of the enemy from the
        given gameState. You must first place the enemy in the gameState, using
        setGhostPosition below.
        """
        tem = self.setEnemyPosition(gameState, pos, self.index)
        dist = DiscreteDistribution()
        actions = tem.getLegalActions(self.index)
        for action in actions:
            pos = tem.generateSuccessor(self.index, action).getAgentPosition(self.index)
            dist[pos] += 1.0
        dist.normalize()
        return dist

    def getObservationProb(self, noisyDistance, pacmanPosition, enemyPosition, gameState):
        """
        Return the probability P(noisyDistance | pacmanPosition, enemyPosition).
        """
        trueDistance = util.manhattanDistance(pacmanPosition, enemyPosition)
        p = gameState.getDistanceProb(trueDistance, noisyDistance)
        return p

    def setEnemyPosition(self, gameState, enemyPosition, index):
        """
        Set the position of the enemy for this inference module to the specified
        position in the supplied gameState.

        Note that calling setEnemyPositions does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code here only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the enemy
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        result = gameState.deepCopy()
        conf = game.Configuration(enemyPosition, game.Directions.STOP)
        result.data.agentStates[index] = game.AgentState(conf, False)
        return result

    def observe(self, index, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getAgentDistances()
        self.observeUpdate(index, distances[self.index], gameState)

    def initialize(self):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.beliefs = DiscreteDistribution()
        self.beliefs[self.initialAgentPosition] = 1.0

    ######################################
    # Methods that need to be overridden #
    ######################################

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError

class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """

    def observeUpdate(self, index, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        if not gameState.getAgentPosition(self.index) == None:
            self.beliefs = DiscreteDistribution()
            self.beliefs[gameState.getAgentPosition(self.index)] = 1.0
            return
        for p in self.legalPositions:
            newP = self.getObservationProb(observation, gameState.getAgentPosition(index), p, gameState)
            self.beliefs[p] = self.beliefs[p] * newP
        self.beliefs.normalize()
        if self.beliefs.total() == 0:
            self.initialize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        successor = DiscreteDistribution()
        for p1 in self.legalPositions:
            if self.beliefs[p1] == 0:
                continue
            newPosDist = self.getPositionDistribution(gameState, p1)
            for p2 in self.legalPositions:
                if newPosDist[p2] == 0:
                    continue
                successor[p2] += self.beliefs[p1] * newPosDist[p2]
        successor.normalize()
        self.beliefs = successor

    def getBeliefDistribution(self):
        return self.beliefs

class DecisionTree:
    trees = {}
    def __init__(self, index, gameState, red):
        # if (index, gameState) in DecisionTree.trees:
        #     self = DecisionTree.trees[(index, gameState)]
        # else:
            self.index = index
            self.nextIndex = index + 1
            if self.index == 3:
                self.nextIndex = 0
            self.gameState = gameState
            self.actions = gameState.getLegalActions(self.index)
            self.subTrees = []
            self.red = red
            # DecisionTree.trees[(index, gameState)] = self

    def construct(self, n):
        if n <= 0:
            return
        if self.subTrees == []:
            for action in self.actions:
                successor = self.gameState.generateSuccessor(self.index, action)
                subTree = DecisionTree(self.nextIndex, successor, not self.red)
                self.subTrees.append(subTree)
        for subTree in self.subTrees:
                subTree.construct(n - 1)

    def getScore(self, caller):
        if self.subTrees == []:
            if caller == None:
                return self.gameState.getScore()
            else:
                return caller.evaluate(self.gameState)
        else:
            if self.red:
                return max([subTree.getScore(caller) for subTree in self.subTrees])
            else:
                return min([subTree.getScore(caller) for subTree in self.subTrees])

    def bestAction(self, caller = None):
        if self.subTrees == []:
            self.construct(1)
        scores = [subTree.getScore(caller) for subTree in self.subTrees]
        if self.red:
            bestScore = max(scores)
        else:
            bestScore = min(scores)
        bestActions = [a for a, s in zip(self.actions, scores) if s == bestScore]
        return random.choice(bestActions)

class QLearningAgent(CaptureAgentWithTools):
    def _chooseAction(self, gameState):
        if not self.lastState == None:
            reward = self.reward(self.lastState, self.lastAction, self.beliefGameState)
            self.update(self.lastState, self.lastAction, gameState, reward)
        legalActions = gameState.getLegalActions(self.index)
        action = None
        if not len(legalActions) == 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(gameState)
        self.doAction(self.beliefGameState, action)
        print(self.index)
        print(self.weights)
        return action

    def reward(self, oldState, newState):
        return 0.0

    def computeValueFromQValues(self, state):
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 0:
            return 0.0
        QValues = [self.getQValue(state, action) for action in legalActions]
        maxQValue = max(QValues)
        return maxQValue

    def computeActionFromQValues(self, state):
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 0:
            return None
        QValues = [self.getQValue(state, action) for action in legalActions]
        maxQValue = max(QValues)
        action = random.choice([action for action, QValue in zip(legalActions, QValues) if QValue == maxQValue])
        return action

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        return self.weights * self.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        features = self.getFeatures(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]
        self.weights.normalize()

    def doAction(self,state,action):
        self.lastState = state
        self.lastAction = action

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    # def getFeatures(self, state, action):
    #     """
    #       Returns a dict from features to counts
    #       Usually, the count will just be 1.0 for
    #       indicator functions.
    #     """
    #     features = util.Counter()
    #     """
    #     closestFood
    #     bias
    #     "#-of-ghosts-1-step-away"
    #     "eats-food"
    #     ghosts
    #     pacman
    #     scaredTime
    #     carryfood
    #     """
    #     return features

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        opponents = self.getOpponents(state)
        food = self.getFood(state)
        walls = state.getWalls()
        ghosts = [self.beliefGameState.getAgentPosition(i) for i in opponents]
        myState = state.getAgentState(self.index)

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        if myState.isPacman:
            features["isPacman"] = 1.0

        features["scaredTime"] = myState.scaredTimer

        features["numCarrying"] = myState.numCarrying

        enemies = [self.beliefGameState.getAgentState(i) for i in opponents]
        features["closestEnemy"] = min([self.getMazeDistance((next_x, next_y), enemy.getPosition()) for enemy in enemies])

        features["hutong"] = self.hutongMap.value((next_x, next_y))

        features.divideAll(10.0)
        return features

class QLearningAgentO(QLearningAgent):
    def otherRegister(self):
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = 100
        self.epsilon = 0.1
        self.alpha = 0.5
        self.discount = 1
        self.weights = util.Counter(
            {'bias': 0.0, '#-of-ghosts-1-step-away': -100.0, 'closest-food': 1.0, 'scaredTime': 0.0, 'numCarrying': 0.0, 'closestEnemy': -0.5, 'hutong': 0.0}
            )
        self.startEpisode()

    def reward(self, oldState, action, newState):
        reward = 0.0
        reward += 10*(self.getScore(newState) - self.getScore(oldState))
        reward += len(self.getFood(newState).asList()) - len(self.getFood(oldState).asList())
        agentState = newState.getAgentState(self.index)
        if agentState.getPosition() == agentState.start.pos:
            reward += -1000
        if action == Directions.STOP:
            reward += -1
        return reward

class QLearningAgentD(QLearningAgent):
    def otherRegister(self):
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = 100
        self.epsilon = 0.1
        self.alpha = 0.5
        self.discount = 1
        self.weights = util.Counter(
            {'bias': 0.0, '#-of-ghosts-1-step-away': 1.0, 'closest-food': 0.0, 'scaredTime': 0.0, 'numCarrying': 0.0, 'closestEnemy': 0.606343778451463, 'hutong': 0.0}
            )

        self.startEpisode()

    def reward(self, oldState, action, newState):
        reward = 0.0
        oldOpponentPos = [oldState.getAgentState(i).getPosition() for i in self.getOpponents(oldState)]
        myPos = newState.getAgentState(self.index).getPosition()
        if myPos in oldOpponentPos:
            reward += 100
        else:
            for a in oldOpponentPos:
                if self.hutongMap.isOnExit(myPos, a):
                    reward += 500
        if myPos == newState.getAgentState(self.index).start.pos:
            reward += -1000
        if action == Directions.STOP:
            reward += -1
        return reward

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None
