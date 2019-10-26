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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'TestHeuristicAgentO', second = 'TestHeuristicAgentD'):
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
oPosList = []
dPosList = []
class TestHeuristicAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.hutongMap = HuTong(gameState.getWalls())
        for i in self.getOpponents(gameState):
            inference[i] = ExactInference(i, gameState)
            inference[i].initialize()

    def chooseAction(self, gameState):
        self.updateBeliefs(gameState)
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, action) for action in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def updateBeliefs(self, gameState):
        if self.index == 0:
            if not self.getPreviousObservation() == None:
                inference[3].elapseTime(gameState)
        else:
            inference[self.index - 1].elapseTime(gameState)
        for i in inference:
            inference[i].observe(self.index, gameState)
        # self.debugDraw(inference[1].getBeliefDistribution().argMax(), (1,0,0), True)
        # self.debugDraw(inference[3].getBeliefDistribution().argMax(), (0,1,0))

class TestHeuristicAgentO(TestHeuristicAgent):
    def evaluate(self, gameState, action):
        score = 0
        successor = gameState.generateSuccessor(self.index, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(successor).asList()
        capsuleList = self.getCapsules(successor)
        oPosList.append(myPos)
        # print("offensiveAgentPositionList:", oPosList)
        if len(oPosList) > 100:
            if oPosList[-1] == oPosList[-3] == oPosList[-5]:
                score += 10000
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
            if not capsuleList == []:
                minCapsuleDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
                print('capsuledistance:',minCapsuleDistance)
                score += -50 * minCapsuleDistance
            score += -0.7 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
        elif len(foodList) > 0:
            score += -100 * len(foodList)
            minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            score += -minFoodDistance
            score += -0.14 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
            if not capsuleList == []:
                minCapsuleDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
                score += -0.5 * minCapsuleDistance
        score += -200 * len([a for a in enemies if a.isPacman or a.scaredTimer > 0 and a.getPosition() != None])
        if action == Directions.STOP:
            score += -10
        score += -0.001 * myState.numCarrying * self.getMazeDistance(myPos, self.start)
        return score

simulateHistory = {}
recordDeep = 15
simulateNumber = 30
simulateDeep = 15
class TestHeuristicAgentOWithMCTS(TestHeuristicAgentO):
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if not defenders == []:
            values = [self.MCTSevaluate(gameState.generateSuccessor(self.index, action)) for action in actions]
        else:
            values = [self.evaluate(gameState.generateSuccessor(self.index, action)) for action in actions]
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
            result = self.evaluate(gameState) - 1000
            if simulateDeep - n < recordDeep:
                self.update(gameState, result)
            return result
        if n == 0:
            return self.evaluate(gameState)
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
        myPos = myState.getPosition()
        dPosList.append(myPos)
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
# self.walls = gameState.getWalls()
# map = HuTong(self.walls)
# path = map.huTongPath
# print(map.branchHuTongList)
# print(map.huTongList, len(map.huTongList))
# print(map.map)
# print(len(path))
# print(map.isOnExit((6,9), (7,9)))
# print(map.value((1,1)))

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
