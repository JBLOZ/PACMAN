# multiAgents.py
# --------------
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

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util
from functools import partial
import math
random.seed(42)  # For reproducibility
from game import Agent
from pacman import GameState


def mazeDistance(point1, point2, gameState):
    """Distancia mínima en el laberinto entre dos puntos usando BFS."""
    from util import Queue
    walls = gameState.getWalls()
    if point1 == point2:
        return 0
    fringe = Queue()
    visited = set()
    fringe.push((point1, 0))
    visited.add(point1)
    while not fringe.isEmpty():
        position, dist = fringe.pop()
        x, y = position
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                if (nx, ny) == point2:
                    return dist + 1
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    fringe.push(((nx, ny), dist + 1))
    return float('inf')

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distToPacman = partial(manhattanDistance, newPos)
        
        def ghostF(ghost):
            dist = distToPacman(ghost.getPosition())
            if ghost.scaredTimer > dist:
                return float('inf')
            if dist <= 1:
                return float('-inf')
            return 0
        ghostScore = min(map(ghostF, newGhostStates))
        
        distToClosestFood = min(map(distToPacman, newFood.asList()), default=float('inf'))
        closestFoodFeature = 1.0 / (1.0 + distToClosestFood)
        return successorGameState.getScore() + closestFoodFeature + ghostScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent implementation for multi-agent Pacman.
    
    This agent uses the minimax algorithm to make optimal decisions in an adversarial
    environment where Pacman (maximizing player) competes against multiple ghosts 
    (minimizing players).
    
    Game Structure:
    - One ply = Pacman move + all ghost moves
    - Depth increments only when control returns to Pacman
    - Agents take turns: Pacman (0) → Ghost1 (1) → Ghost2 (2) → ... → Pacman (depth++)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        
        The algorithm guarantees optimal play assuming all agents play optimally:
        - Pacman maximizes the evaluation score
        - Ghosts minimize Pacman's score
        """
        
        def minimax(agentIndex, depth, gameState):
            """
            Recursive minimax function that evaluates game states.
            
            Args:
                agentIndex: Current agent (0=Pacman, 1+=Ghosts)
                depth: Current depth in the game tree
                gameState: Current state of the game
                
            Returns:
                Best evaluation score for this state
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (Maximizing player)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghost's turn (Minimizing player)
            else:
                return minValue(agentIndex, depth, gameState)
        
        def maxValue(agentIndex, depth, gameState):
            """
            Handles Pacman's moves (maximizing player).
            Chooses the action that leads to the highest evaluation score.
            """
            v = float('-inf')  # Start with worst possible value for Pacman
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available - return current evaluation
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Try each possible action and choose the best
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # After Pacman moves, first ghost plays (agent 1)
                v = max(v, minimax(1, depth, successor))
            return v

        def minValue(agentIndex, depth, gameState):
            """
            Handles Ghost moves (minimizing players).
            Chooses the action that leads to the lowest evaluation score for Pacman.
            """
            v = float('inf')  # Start with best possible value for Pacman
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available - return current evaluation
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Determine next agent and depth
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If all ghosts have moved, return to Pacman and increment depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0      # Back to Pacman
                nextDepth = depth + 1  # New ply begins

            # Try each possible action and choose the worst for Pacman
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, nextDepth, successor))
            return v

        # Main decision logic for Pacman
        bestAction = None
        bestScore = float('-inf')

        # Get all legal actions for Pacman
        legalActions = gameState.getLegalActions(0)
        
        # Handle edge case: no legal actions
        if not legalActions:
            return Directions.STOP

        # Try each legal action for Pacman and find the best
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # Start minimax with first ghost (agent 1) at current depth
            score = minimax(1, 0, successor)
            
            # Update best action if this score is better
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction if bestAction is not None else Directions.STOP

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning optimization.
    
    This agent uses the alpha-beta pruning technique to optimize the minimax algorithm
    by eliminating branches that cannot possibly influence the final decision.
    
    Alpha-Beta Pruning maintains two values:
    - Alpha (α): The best value the maximizer (Pacman) has found so far
    - Beta (β): The best value the minimizer (Ghosts) has found so far
    
    Pruning conditions:
    - In MAX nodes: If value ≥ β, prune remaining children
    - In MIN nodes: If value ≤ α, prune remaining children
    
    This provides the exact same result as regular minimax but with significantly
    improved computational efficiency.
    """

    def getAction(self, gameState: GameState):
        """
        Returns the alpha-beta action using self.depth and self.evaluationFunction.
        
        The algorithm explores the game tree with alpha-beta bounds to prune
        branches that cannot affect the optimal decision.
        """
        
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            """
            Core alpha-beta search function.
            
            Args:
                agentIndex: Current agent (0=Pacman, 1+=Ghosts)
                depth: Current search depth
                gameState: Current game state
                alpha: Best value for maximizer so far
                beta: Best value for minimizer so far
            
            Returns:
                Best evaluation score for the current state
            """
            # Base case: Terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Dispatch to appropriate player function
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            """
            Handles Pacman's moves (maximizing player) with alpha-beta pruning.
            
            Tries to maximize the evaluation score while updating alpha
            and pruning when value ≥ beta.
            """
            v = float('-inf')  # Start with worst possible value for maximizer
            legalActions = gameState.getLegalActions(agentIndex)

            # No legal actions available - return current evaluation
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Try each possible action
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # After Pacman moves, first ghost plays (agent 1)
                v = max(v, alphabeta(1, depth, successor, alpha, beta))
                
                # Alpha-beta pruning: if v ≥ β, remaining actions won't be chosen by minimizer
                if v > beta:
                    return v  # Prune remaining branches
                
                # Update alpha with best value found so far
                alpha = max(alpha, v)
            
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            """
            Handles Ghost moves (minimizing players) with alpha-beta pruning.
            
            Tries to minimize Pacman's score while updating beta
            and pruning when value ≤ alpha.
            """
            v = float('inf')  # Start with best possible value for maximizer
            legalActions = gameState.getLegalActions(agentIndex)

            # No legal actions available - return current evaluation
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Determine next agent and depth
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If all ghosts have moved, return to Pacman and increment depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0      # Back to Pacman
                nextDepth = depth + 1  # New ply begins

            # Try each possible action
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphabeta(nextAgent, nextDepth, successor, alpha, beta))
                
                # Alpha-beta pruning: if v ≤ α, remaining actions won't be chosen by maximizer
                if v < alpha:
                    return v  # Prune remaining branches
                
                # Update beta with worst value found so far (from maximizer's perspective)
                beta = min(beta, v)
            
            return v

        # Choose the best action for Pacman at the root level
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # Try each possible action for Pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start alpha-beta search with first ghost (agent 1)
            score = alphabeta(1, 0, successor, alpha, beta)
            
            # Update best action if this score is better
            if score > bestScore:
                bestScore = score
                bestAction = action
            
            # Update alpha for future iterations
            alpha = max(alpha, score)

        return bestAction if bestAction is not None else Directions.STOP

        
#############################################
# ExpectimaxAgent (q4) y mejor heurística (q5)
#############################################

################ Expectimax mejorado ################

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        maxValue = -1e9
        maxAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            sucState = gameState.generateSuccessor(0, action)
            sucValue = self.getNodeValue(sucState, 0, 1)
            if sucValue > maxValue:
                maxValue, maxAction = sucValue, action

        return maxAction

    def getNodeValue(self, gameState, curDepth, agentIndex):
        if curDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maxValue(gameState, curDepth)
        else:
            return self.expValue(gameState, curDepth, agentIndex)

    def maxValue(self, gameState, curDepth):
        maxValue = -1e9
        for action in gameState.getLegalActions(0):
            sucState = gameState.generateSuccessor(0, action)
            sucValue = self.getNodeValue(sucState, curDepth, 1)
            maxValue = max(maxValue, sucValue)
        return maxValue

    def expValue(self, gameState, curDepth, agentIndex):
        totalValue = 0.0
        numAgent = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            sucState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numAgent - 1:
                sucValue = self.getNodeValue(sucState, curDepth + 1, 0)
            else:
                sucValue = self.getNodeValue(sucState, curDepth, agentIndex + 1)
            totalValue += sucValue

        numAction = len(gameState.getLegalActions(agentIndex))
        return totalValue / numAction


############# Heurística: betterEvaluationFunction #############

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Consts
    INF = 100000000.0  # Infinite value
    WEIGHT_FOOD = 10.0  # Food base value
    WEIGHT_GHOST = -10.0  # Ghost base value
    WEIGHT_SCARED_GHOST = 100.0  # Scared ghost base value

    # Base on gameState.getScore()
    score = currentGameState.getScore()

    # Evaluate the distance to the closest food
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    if len(distancesToFoodList) > 0:
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        score += WEIGHT_FOOD

    # Evaluate the distance to ghosts
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:  # If scared, add points
                score += WEIGHT_SCARED_GHOST / distance
            else:  # If not, decrease points
                score += WEIGHT_GHOST / distance
        else:
            return -INF  # Pacman is dead at this point

    return score

# Alias para autograder
better = betterEvaluationFunction


        

###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()
        
        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
        
        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    score -= 200  # Gran penalización por estar demasiado cerca
        
        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100
        
        return score + neural_score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)