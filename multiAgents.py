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

# Global seed variable for reproducibility


import torch
import numpy as np

from net import PacmanNet, get_map_matrix_from_gamestate 
from net import IDX_TO_ACTION, HIDDEN_SIZE_FC, NUM_ACTIONS as NN_OUTPUT_ACTIONS

import os
from util import manhattanDistance
from game import Directions
import random, util
from functools import partial
import math
random.seed(42)  # Use the global seed variable for reproducibility
from game import Agent
from pacman import GameState



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
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Agente Minimax con poda alfa-beta mejorado con ordenación neural de movimientos.
    
    Este agente implementa el algoritmo minimax con poda alfa-beta para búsqueda adversarial,
    con una mejora clave: utiliza una red neuronal entrenada (PacmanNet) para ordenar los
    movimientos de Pacman de manera inteligente. Esto mejora significativamente la eficiencia
    de la poda alfa-beta al explorar primero los movimientos más prometedores.
    
    Características principales:
    - Implementa minimax con poda alfa-beta estándar
    - Usa una red neuronal para predecir la calidad de los movimientos de Pacman
    - Ordena los movimientos por probabilidad antes de explorarlos (move ordering)
    - Mantiene compatibilidad total con el framework de multiAgents
    """

    def __init__(self, evalFn='evaluationFunction', depth='2',
                 action_nn_path="models/pacman_model_cnn_final.pth",
                 nn_input_dims_str="20x11", device_name='cpu'):
        """
        Inicializa el agente AlphaBeta con capacidades de red neuronal.
        
        Args:
            evalFn (str): Nombre de la función de evaluación para nodos hoja.
                         Por defecto usa 'evaluationFunction' que está definida más abajo.
            depth (str): Profundidad máxima de búsqueda para Pacman (número de movimientos
                        que Pacman puede planificar hacia adelante).
            action_nn_path (str): Ruta al archivo .pth que contiene la red neuronal entrenada.
                                 Esta red predice qué acciones son más prometedoras.
            nn_input_dims_str (str): Dimensiones que espera la red neuronal en formato "alto x ancho".
                                   Debe coincidir con cómo fue entrenada la red.
            device_name (str): Dispositivo para PyTorch ('cpu' o 'cuda').
        
        La inicialización hace tres cosas principales:
        1. Configura el agente multiagente básico (función de evaluación y profundidad)
        2. Carga y prepara la red neuronal para ordenar movimientos
        3. Establece el mapeo entre índices de la red y acciones del juego
        """
       
        # Llamar al constructor padre para configurar self.evaluationFunction y self.depth
        super().__init__(evalFn, depth)
        
        # Configuración del dispositivo PyTorch (CPU vs GPU)
        self.device = torch.device(device_name)
        
        # Parsear las dimensiones de entrada de la red neuronal
        # Ejemplo: "20x11" -> (20, 11) que representa (alto, ancho) del mapa
        dims = tuple(map(int, nn_input_dims_str.split('x')))
        self.nn_input_dims = dims
        
        # Mapeo de índices de la red neuronal a acciones del juego
        # IDX_TO_ACTION mapea 0->North, 1->South, etc.
        self.idx_to_action_map = IDX_TO_ACTION 

        # Cargar la red neuronal entrenada
        # Esta red toma el estado del juego como matriz y predice probabilidades para cada acción
        self.action_predictor_nn = PacmanNet(self.nn_input_dims, HIDDEN_SIZE_FC, NN_OUTPUT_ACTIONS).to(self.device)
        model_data = torch.load(action_nn_path, map_location=self.device)
        self.action_predictor_nn.load_state_dict(model_data['model_state_dict'])
        self.action_predictor_nn.eval()  # Modo evaluación (sin entrenamiento)

    def _get_ordered_pacman_actions(self, gameState, legalActions):
        """
        Ordena las acciones legales de Pacman usando predicciones de la red neuronal.
        
        Esta es la función clave que diferencia nuestro agente de un alpha-beta estándar.
        La ordenación de movimientos (move ordering) es crucial para la eficiencia de alpha-beta:
        si exploramos primero los mejores movimientos, obtendremos podas más tempranas y
        reduciremos drásticamente el número de nodos explorados.
        
        Args:
            gameState: Estado actual del juego de Pacman
            legalActions: Lista de acciones legales disponibles para Pacman
            
        Returns:
            list: Acciones ordenadas de mejor a peor según la red neuronal.
                 Si hay algún error con la red, devuelve las acciones originales.
                 
        Proceso paso a paso:
        1. Convierte el estado del juego a una matriz numérica que entiende la red
        2. Ejecuta la red neuronal para obtener probabilidades de cada acción
        3. Mapea estas probabilidades a las acciones legales específicas
        4. Ordena las acciones por probabilidad descendente (mejor primero)
        """

        try:
            # Paso 1: Convertir el estado del juego a matriz numérica
            # get_map_matrix_from_gamestate convierte paredes, comida, fantasmas, etc.
            # en una representación numérica que puede procesar la red neuronal
            map_matrix = get_map_matrix_from_gamestate(gameState, self.nn_input_dims)
            
            # Paso 2: Preparar tensor para PyTorch
            # La red espera formato (batch, channels, height, width) o (batch, height, width)
            map_tensor = torch.FloatTensor(map_matrix).to(self.device)
            if len(map_tensor.shape) == 2:  # Si es (height, width)
                map_tensor = map_tensor.unsqueeze(0)  # Añadir dimensión batch -> (1, height, width)

            # Paso 3: Ejecutar la red neuronal sin calcular gradientes
            with torch.no_grad():
                # La red devuelve logits (puntuaciones brutas) para cada acción posible
                action_logits = self.action_predictor_nn(map_tensor).squeeze() 

            # Paso 4: Mapear las predicciones de la red a acciones del juego
            action_scores = {}
            
            
            for i in range(NN_OUTPUT_ACTIONS):
                # Convertir índice numérico a nombre de acción (ej: 0 -> 'North')
                action_name_from_nn = self.idx_to_action_map.get(i)
                if action_name_from_nn:
                    # Guardar la puntuación que la red asignó a esta acción
                    action_scores[action_name_from_nn] = action_logits[i].item()
            

            # Paso 5: Ordenar acciones legales por puntuación de la red (mejor primero)
            sorted_legal_actions = sorted(legalActions,
                                          key=lambda action: action_scores.get(action, -float('inf')),
                                          reverse=True)  # Mayor score primero
            return sorted_legal_actions
            
        except Exception as e:
            # Manejo robusto de errores: si algo falla, usar orden original
            print(f"Error (AlphaBetaAgent) durante la ordenación de movimientos con NN: {e}. Usando orden por defecto.")
            return legalActions
        
    def getAction(self, gameState: GameState):
        """
        Método principal que implementa el algoritmo minimax con poda alfa-beta.
        
        Este método coordina toda la búsqueda adversarial. La estructura sigue el
        patrón estándar de minimax para juegos con múltiples agentes:
        - Pacman (agente 0) es el maximizador
        - Los fantasmas (agentes 1, 2, ...) son minimizadores
        - La búsqueda alterna entre estos agentes hasta alcanzar la profundidad máxima
        
        Mejoras implementadas:
        1. Poda alfa-beta para reducir el espacio de búsqueda
        2. Ordenación neural de movimientos para Pacman (mejora la eficiencia de la poda)
        3. Manejo robusto de casos límite y estados terminales
        
        Returns:
            str: La mejor acción encontrada (ej: 'North', 'South', etc.)
        """
        
        # =================== FUNCIONES INTERNAS DE BÚSQUEDA ===================
        # Estas funciones implementan la lógica recursiva del algoritmo minimax
        
        def perform_alphabeta_search(agentIndex, current_search_depth, current_gameState, alpha, beta):
            """
            Función recursiva principal que implementa minimax con poda alfa-beta.
            
            Args:
                agentIndex (int): Qué agente está jugando (0=Pacman, 1+=fantasmas)
                current_search_depth (int): Profundidad actual de búsqueda para Pacman
                current_gameState: Estado actual del juego
                alpha (float): Mejor valor encontrado para el maximizador
                beta (float): Mejor valor encontrado para el minimizador
                
            Returns:
                float: Valor de utilidad del estado evaluado
                
            Lógica de profundidad:
            - current_search_depth cuenta solo los movimientos de Pacman
            - Cuando current_search_depth == self.depth, Pacman ha planificado suficiente
            - Los fantasmas no incrementan la profundidad, solo Pacman
            """
            # Caso base 1: Estados terminales (victoria/derrota)
            if current_gameState.isWin() or current_gameState.isLose():
                return self.evaluationFunction(current_gameState)
            
            if agentIndex == 0:  # Turno de Pacman (Maximizador)
                # Caso base 2: Pacman ha alcanzado su profundidad máxima de planificación
                if current_search_depth == self.depth:
                     return self.evaluationFunction(current_gameState)
                return get_max_value(agentIndex, current_search_depth, current_gameState, alpha, beta)
            else:  # Turno de un Fantasma (Minimizador)
                return get_min_value(agentIndex, current_search_depth, current_gameState, alpha, beta)

        def get_max_value(agentIndex, current_search_depth, current_gameState, alpha, beta):
            """
            Implementa la lógica de maximización para Pacman.
            
            Esta función busca el movimiento que maximiza la utilidad para Pacman.
            Aquí es donde aplicamos la ordenación neural de movimientos para mejorar
            la eficiencia de la poda alfa-beta.
            
            Args:
                agentIndex (int): Debe ser 0 (Pacman)
                current_search_depth (int): Profundidad actual
                current_gameState: Estado del juego
                alpha, beta (float): Ventanas de poda alfa-beta
                
            Returns:
                float: Máximo valor de utilidad encontrado
            """
            v = float('-inf')  # Inicializar con el peor valor posible
            legalActions = current_gameState.getLegalActions(agentIndex)

            # Caso especial: si no hay acciones legales, evaluar estado actual
            if not legalActions:
                return self.evaluationFunction(current_gameState)

            # *** APLICACIÓN DE ORDENACIÓN NEURAL ***
            # Esta es la mejora clave: ordenar los movimientos por probabilidad neural
            # antes de explorarlos. Esto mejora dramáticamente la eficiencia de alpha-beta.
            ordered_actions = self._get_ordered_pacman_actions(current_gameState, legalActions)

            # Explorar cada acción en el orden optimizado
            for action in ordered_actions:
                successor = current_gameState.generateSuccessor(agentIndex, action)
                
                # Después de que Pacman se mueve, le toca al primer fantasma (agente 1)
                # La profundidad no se incrementa aquí porque Pacman aún no ha completado su "turno"
                v = max(v, perform_alphabeta_search(1, current_search_depth, successor, alpha, beta))
                
                # *** PODA ALFA-BETA ***
                if v > beta:
                    return v  # Poda beta: el minimizador nunca eligirá esta rama
                alpha = max(alpha, v)  # Actualizar la mejor opción del maximizador
            return v

        def get_min_value(agentIndex, current_search_depth, current_gameState, alpha, beta):
            """
            Implementa la lógica de minimización para los fantasmas.
            
            Los fantasmas intentan minimizar la utilidad de Pacman. Esta función
            maneja la rotación entre múltiples fantasmas y el retorno a Pacman.
            
            Args:
                agentIndex (int): Índice del fantasma actual (>=1)
                current_search_depth (int): Profundidad actual
                current_gameState: Estado del juego
                alpha, beta (float): Ventanas de poda alfa-beta
                
            Returns:
                float: Mínimo valor de utilidad encontrado
            """
            v = float('inf')  # Inicializar con el peor valor posible para el minimizador
            legalActions = current_gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(current_gameState)

            # Determinar qué agente juega después
            nextAgent = agentIndex + 1
            next_search_depth_for_pacman = current_search_depth 

            # Si todos los fantasmas se han movido, vuelve a Pacman
            if nextAgent == current_gameState.getNumAgents():
                nextAgent = 0  # Volver a Pacman
                next_search_depth_for_pacman = current_search_depth + 1  # Incrementar profundidad de Pacman
            
            # Explorar cada acción del fantasma
            for action in legalActions:
                successor = current_gameState.generateSuccessor(agentIndex, action)
                v = min(v, perform_alphabeta_search(nextAgent, next_search_depth_for_pacman, successor, alpha, beta))
                
                # *** PODA ALFA-BETA ***
                if v < alpha:
                    return v  # Poda alfa: el maximizador nunca eligirá esta rama
                beta = min(beta, v)  # Actualizar la mejor opción del minimizador
            return v

        # =================== LÓGICA PRINCIPAL DE DECISIÓN ===================
        
        # Variables para rastrear la mejor acción encontrada
        bestAction = None
        bestScore = float('-inf')
        alpha_root = float('-inf')  # Ventana alfa para el nodo raíz
        beta_root = float('inf')    # Ventana beta para el nodo raíz

        # Obtener acciones legales para Pacman en el estado actual
        pacman_legal_actions = gameState.getLegalActions(0)
        
        # *** APLICAR ORDENACIÓN NEURAL AL NIVEL RAÍZ ***
        # Esto es crucial: ordenar las acciones iniciales de Pacman por probabilidad
        # para explorar primero las más prometedoras
        ordered_initial_actions = self._get_ordered_pacman_actions(gameState, pacman_legal_actions)

        # Evaluar cada acción posible de Pacman
        for action in ordered_initial_actions:
            # Generar el estado resultante de esta acción
            successor = gameState.generateSuccessor(0, action)
            
            # Iniciar la búsqueda minimax desde el primer fantasma (agente 1)
            # La profundidad inicial es 0 porque Pacman aún no ha completado su primer "nivel"
            score = perform_alphabeta_search(1, 0, successor, alpha_root, beta_root)
            
            # Rastrear la mejor acción encontrada hasta ahora
            if score > bestScore:
                bestScore = score
                bestAction = action
            
            # Actualizar alfa para mejorar las podas futuras
            alpha_root = max(alpha_root, score)        # Retornar la mejor acción, o STOP como último recurso
        return bestAction if bestAction is not None else Directions.STOP

    """
    =================== RESUMEN DEL ALGORITMO ===================
    Flujo de ejecución:
    1. Obtener acciones legales de Pacman
    2. Ordenar estas acciones usando la red neuronal
    3. Para cada acción (en orden de calidad):
       a. Generar estado sucesor
       b. Iniciar búsqueda minimax desde los fantasmas
       c. Aplicar poda alfa-beta para eficiencia
    4. Retornar la acción con mejor evaluación
    """

def evaluationFunction(currentGameState: GameState):
    """
    Función de evaluación de AlphaBeta Pacman.
    
    Esta función evalúa qué tan "bueno" es un estado de juego para Pacman,
    considerando múltiples factores críticos que influyen en el éxito:
    - Puntuación actual del juego
    - Proximidad a la comida
    - Proximidad y estado de los fantasmas
    
    La función utiliza un enfoque de suma ponderada donde cada factor contribuye
    al puntaje final según su importancia estratégica.
    
    Returns:
        float: Puntaje de evaluación del estado. Valores más altos indican
               estados más favorables para Pacman.
    """
    
    # =================== EXTRACCIÓN DE INFORMACIÓN DEL ESTADO ===================
    # Obtener los elementos clave del estado actual del juego
    newPos = currentGameState.getPacmanPosition()      # Posición actual de Pacman (x, y)
    newFood = currentGameState.getFood()               # Grid booleano con posiciones de comida
    newGhostStates = currentGameState.getGhostStates() # Lista de estados de todos los fantasmas

    # =================== DEFINICIÓN DE PESOS ESTRATÉGICOS ===================
    # Estos pesos determinan la importancia relativa de cada factor en la evaluación.
    # Han sido ajustados experimentalmente para obtener el mejor comportamiento.
    
    WEIGHT_FOOD = 12.0          # Peso para proximidad a comida
                                # Valor positivo: queremos estar cerca de la comida
                                # Moderado: importante pero no crítico
    
    WEIGHT_GHOST = -30.0        # Peso para proximidad a fantasmas normales  
                                # Valor negativo: queremos evitar fantasmas peligrosos
                                # Igual magnitud que comida: equilibra ataque/defensa
    
    WEIGHT_SCARED_GHOST = 150.0 # Peso para proximidad a fantasmas asustados
                                # Valor muy alto: cazar fantasmas asustados da muchos puntos
                                # 10x mayor que comida: prioriza oportunidades de caza

    # =================== EVALUACIÓN BASE: PUNTUACIÓN DEL JUEGO ===================
    # Comenzamos con la puntuación oficial del juego, multiplicada por 5 para darle peso
    # La puntuación incluye: comida consumida, fantasmas cazados, bonificaciones de tiempo
    score = currentGameState.getScore() * 5
    # Razón del multiplicador x5: 
    # - Multiplicar por 5 asegura que la puntuación base tenga influencia significativa

    # =================== EVALUACIÓN DE PROXIMIDAD A LA COMIDA ===================
    # Estrategia: Incentivar a Pacman a acercarse a la comida más cercana
    # Usamos distancia manhattan porque Pacman se mueve en rejilla (no diagonal)
    
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    
    if len(distancesToFoodList) > 0:
        # Hay comida disponible: usar distancia inversa para priorizar comida cercana
        # Formula: PESO / distancia_mínima
        # - Distancia 1: +10 puntos (muy bueno, comida al lado)
        # - Distancia 5: +2 puntos (moderado, comida a distancia media)
        # - Distancia 10: +1 punto (poco atractivo, comida lejana)
        score += WEIGHT_FOOD / min(distancesToFoodList)
    else:
        # No hay comida restante: bonificación completa (probablemente estado de victoria)
        # En este caso, hemos comido toda la comida, lo cual es excelente
        score += WEIGHT_FOOD

    # =================== EVALUACIÓN DE PROXIMIDAD A FANTASMAS ===================
    # Estrategia dual: evitar fantasmas peligrosos, perseguir fantasmas asustados
    # Esta es la parte más compleja porque debe manejar dos comportamientos opuestos
    
    for ghost in newGhostStates:
        # Calcular distancia a este fantasma específico
        distance = manhattanDistance(newPos, ghost.getPosition())
        
        if distance > 0:
            # Pacman no está exactamente en la misma posición que el fantasma
            
            if ghost.scaredTimer > 0:  
                # FANTASMA ASUSTADO: Es una oportunidad de oro
                # Los fantasmas asustados pueden ser cazados para obtener puntos altos
                # Formula: PESO_DEL_FANTASMA / distancia
                # - Distancia 1: +100 puntos (perseguir inmediatamente)
                # - Distancia 2: +50 puntos (sigue siendo muy buena opcion perseguirlo)
                # - Distancia 5: +20 puntos (vale la pena perseguir pero puede haber mejores decisiones)
                score += WEIGHT_SCARED_GHOST / distance
                
                # Razón del peso alto (100):
                # - Cazar un fantasma da 200 puntos en el juego real
                # - Es una oportunidad temporal (scaredTimer se agota)
                # - Debe priorizarse sobre comer comida normal
                
            else:  
                # FANTASMA NORMAL: Es una amenaza mortal
                # Tocar un fantasma normal termina el juego inmediatamente
                # Formula: PESO_NEGATIVO / distancia
                # - Distancia 1: -10 puntos (¡muy peligroso!)
                # - Distancia 2: -5 puntos (peligroso)
                # - Distancia 5: -2 puntos (precaución moderada)
                score += WEIGHT_GHOST / distance
                
                # Razón del peso negativo (-10):
                # - Equilibra con la atracción de la comida (+10)
                # - Evita que Pacman se acerque temerariamente a fantasmas
                # - Permite cierto riesgo calculado cuando la recompensa es alta
                
        else:
            # COLISIÓN DETECTADA: Pacman está exactamente en la posición del fantasma
            # Esto solo puede pasar con un fantasma normal (los asustados serían cazados)
            # Retornamos inmediatamente el peor puntaje posible: muerte segura
            return -float('inf')  
            # Razón del -infinito:
            # - Muerte de Pacman es el peor resultado posible
            # - Debe evitarse a toda costa, sin importar otros factores
            # - Garantiza que ningún camino que lleve a muerte sea elegido

    # =================== RETORNO DEL PUNTAJE FINAL ===================
    # El puntaje final es la suma de todos los factores considerados:
    # = puntuación_base + factor_comida + suma_factores_fantasmas
    return score

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
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