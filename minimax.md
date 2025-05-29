# Game Theory and Adversarial Search

## Practical Session: Minimax Algorithm

### Introduction

In this session, we will explore the concept of adversarial search through the **Minimax algorithm**. This algorithm is fundamental in game theory and is used to make optimal decisions in competitive environments where multiple agents have opposing objectives.

### What is Minimax?

The Minimax algorithm is a decision-making algorithm used in **two-player zero-sum games**. The key principle is that one player (the maximizer) tries to maximize their score, while the opponent (the minimizer) tries to minimize the maximizer's score.

#### Core Concepts

1. **Game Tree**: A tree structure representing all possible game states
2. **Max Player**: The player trying to maximize the evaluation score
3. **Min Player**: The player trying to minimize the evaluation score  
4. **Evaluation Function**: A function that assigns a numeric value to game states
5. **Depth**: How many moves ahead to look in the game tree

#### The Minimax Principle

```
Max Level:  Choose the action that leads to the HIGHEST value
Min Level:  Choose the action that leads to the LOWEST value
```

The algorithm works by recursively evaluating all possible future states, assuming both players play optimally.

### Simple Example: Tic-Tac-Toe

Consider a simplified tic-tac-toe position where it's X's turn:

```
Current state:
X | O |  
---------
  | X |  
---------
  | O |
```

X (maximizer) has three possible moves. The minimax algorithm will:
1. Try each possible move
2. Assume O (minimizer) will respond optimally
3. Choose the move that guarantees the best outcome

### Mathematical Formulation

For a game state `s`, the minimax value is defined as:

```
minimax(s) = {
    evaluation(s)                           if s is terminal
    max(minimax(child) for child in s)      if s is Max node
    min(minimax(child) for child in s)      if s is Min node
}
```

## Application to Pacman

Now we will apply the Minimax algorithm to the Pacman game, where:
- **Pacman (Agent 0)**: Maximizing player - wants to maximize score
- **Ghosts (Agents 1+)**: Minimizing players - want to minimize Pacman's score

### Game Structure in Pacman

In Pacman Minimax with **2 ghosts**:
- One **ply** = Pacman moves + Ghost 1 moves + Ghost 2 moves
- **Depth 1** = One complete ply (Pacman + both ghosts)
- **Depth 2** = Two complete plies

### Agent Order

The agents take turns in this specific order:
1. **Pacman (Agent 0)** - MAX player
2. **Ghost 1 (Agent 1)** - MIN player  
3. **Ghost 2 (Agent 2)** - MIN player
4. Back to **Pacman (Agent 0)** - depth increments

This cycle repeats until the maximum depth is reached.

### Detailed Example: Depth 2 with 2 Ghosts

Let's walk through a complete example with **depth = 2** and **2 ghosts**:

#### Initial Setup

```
Game State:
- Pacman at position (1,1)
- Ghost 1 at position (3,1)
- Ghost 2 at position (1,3)
- Food dot at (2,1)
- Evaluation scores for different scenarios:
  * Pacman eats food: +10
  * Ghost catches Pacman: -500
  * Normal move: -1 (time penalty)
```

#### The Game Tree (Depth 2)

```
                    ROOT: Pacman Turn (MAX)
                   /          |          \
               NORTH        EAST        SOUTH
              Score: ?    Score: ?    Score: ?
                |            |            |
        
    DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)
         /    |    \                  /    |    \                  /    |    \
      N     E     S                N     E     S                N     E     S  
      |     |     |                |     |     |                |     |     |
      
    DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)
         /    |    \                  /    |    \                  /    |    \
      N     E     S                N     E     S                N     E     S  
      |     |     |                |     |     |                |     |     |
      
    DEPTH 1: Pacman Turn (MAX)   DEPTH 1: Pacman Turn (MAX)   DEPTH 1: Pacman Turn (MAX)
    /    |    \                  /    |    \                  /    |    \
   N    E    S                  N    E    S                  N    E    S
   |    |    |                  |    |    |                  |    |    |
   
   [... continues for depth 1 with both ghosts ...]
   
 DEPTH 2: EVALUATE            DEPTH 2: EVALUATE            DEPTH 2: EVALUATE
   +9  -501  +8                +8   +9  -501               -501  +8   +7
```

#### Step-by-Step Execution

**Step 1: Root Node (Pacman's initial turn)**
- Pacman has 3 possible actions: NORTH, EAST, SOUTH
- We need to evaluate each branch

**Step 2: Evaluate EAST branch**

2.1. **Pacman moves EAST** → Pacman at (2,1), eats food (+10 points)

2.2. **Ghost 1's turn (MIN)**
- Ghost 1 can move: NORTH, EAST, SOUTH
- Let's say Ghost 1 moves NORTH → Ghost 1 at (3,0)

2.3. **Ghost 2's turn (MIN)**
- Ghost 2 can move: NORTH, EAST, SOUTH
- Let's say Ghost 2 moves EAST → Ghost 2 at (2,3)

2.4. **Pacman's turn again (depth 1)**
- Pacman can move: NORTH, EAST, SOUTH
- Evaluate each possibility with the new ghost positions:
  - NORTH: Pacman at (2,2) → Continue with both ghosts...
  - EAST: Pacman at (3,1) → Continue with both ghosts...
  - SOUTH: Pacman at (2,0) → Continue with both ghosts...

2.5. **Complete evaluation for this path**
- After all ghost responses and final Pacman moves, this path evaluates to **+5**

2.6. **Back to Ghost 2's choices (step 2.3)**
- Ghost 2 evaluates all its moves:
  - NORTH: Leads to final value +3
  - EAST: Leads to final value +5  ← We calculated this
  - SOUTH: Leads to final value +4
- Ghost 2 chooses MIN: min(+3, +5, +4) = **+3**

2.7. **Back to Ghost 1's choices (step 2.2)**
- Ghost 1 evaluates all its moves (each leading to Ghost 2's turn):
  - NORTH: Leads to Ghost 2 min = +3  ← We calculated this
  - EAST: Leads to Ghost 2 min = +2
  - SOUTH: Leads to Ghost 2 min = +1
- Ghost 1 chooses MIN: min(+3, +2, +1) = **+1**

**Therefore: Pacman EAST → Final value = +1**

**Step 3: Evaluate NORTH branch** 
2.1. **Pacman moves NORTH** → Pacman at (1,2)

2.2. **Ghost 1's turn** → Tries all moves → Best for Ghost 1: +2
2.3. **Ghost 2's turn** → Tries all moves → Best for Ghost 2: +1
2.4. **Continue recursion...** → Final evaluation: **+1**

**Step 4: Evaluate SOUTH branch**
2.1. **Pacman moves SOUTH** → Pacman at (1,0)
2.2. **Ghost 1's turn** → Tries all moves → Best for Ghost 1: -1
2.3. **Ghost 2's turn** → Tries all moves → Best for Ghost 2: -2  
2.4. **Continue recursion...** → Final evaluation: **-2**

**Step 5: Pacman's final decision**
- NORTH = +1
- EAST = +1  ← TIE! (Same as NORTH)
- SOUTH = -2

**Pacman chooses NORTH or EAST** (depends on which is evaluated first) because they both guarantee +1, which is better than -2.

### Implementation for Pacman

```python
class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for Pacman with multiple ghosts
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction.
        """
        
        def minimax(agentIndex, depth, gameState):
            """
            Recursive minimax function
            
            Args:
            - agentIndex: Current agent (0=Pacman, 1+=Ghosts)  
            - depth: Current depth in the game tree
            - gameState: Current state of the game
            
            Returns:
            - Best evaluation score for this state
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (Maximizer)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghost's turn (Minimizer)  
            else:
                return minValue(agentIndex, depth, gameState)
        
        def maxValue(agentIndex, depth, gameState):
            """
            Handles Pacman's moves (maximizing player)
            """
            v = float('-inf')  # Start with worst possible value
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
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
            Handles Ghost moves (minimizing players)
            """
            v = float('inf')  # Start with best possible value for Pacman
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
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

        # Try each legal action for Pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start minimax with first ghost (agent 1) at current depth
            score = minimax(1, 0, successor)
            
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
```

### Complete Execution Trace

Let's trace through our depth-2 example step by step:

**Initial Call**: `getAction(gameState)`

**1. Try Pacman action: EAST**
   - `generateSuccessor(0, EAST)` → New state with Pacman at (2,1)
   - Call `minimax(1, 0, successor_state)`

**2. Ghost's turn (agentIndex=1, depth=0)**
   - `minValue(1, 0, state)` called
   - Ghost legal actions: [NORTH, EAST, SOUTH]
   
   **2.1 Try Ghost 1 action: NORTH**
   - `generateSuccessor(1, NORTH)` → Ghost 1 at (3,0)
   - nextAgent = 2 (Ghost 2), nextDepth = 0
   - Call `minimax(2, 0, successor_state)`
   
   **2.1.1 Ghost 2's turn (agentIndex=2, depth=0)**
   - `minValue(2, 0, state)` called
   - Ghost 2 legal actions: [NORTH, EAST, SOUTH]
   
   **2.1.1.1 Try Ghost 2 action: NORTH**
   - `generateSuccessor(2, NORTH)` → Ghost 2 at (1,4)
   - nextAgent = 3, but numAgents = 3, so nextAgent = 0, nextDepth = 1
   - Call `minimax(0, 1, successor_state)`
   
   **2.1.1.1.1 Pacman's turn (agentIndex=0, depth=1)**
   - `maxValue(0, 1, state)` called
   - Pacman legal actions: [NORTH, EAST, SOUTH]
   - Try each action, call minimax for each with deeper depth...
   - Eventually returns max value: +3
   
   **2.1.1.1 Back to Ghost 2 choice NORTH** → value +3
   
   **2.1.1.2 Try Ghost 2 action: EAST** → Eventually value +5
   
   **2.1.1.3 Try Ghost 2 action: SOUTH** → Eventually value +4
   
   **2.1.1 Ghost 2 chooses minimum**: min(+3, +5, +4) = +3
   
   **2.1 Back to Ghost 1 choice NORTH** → value +3

   **2.2 Try Ghost 1 action: EAST** → Eventually value +2
   
   **2.3 Try Ghost 1 action: SOUTH** → Eventually value +1

**3. Ghost 1 chooses minimum**: min(+3, +2, +1) = +1

**4. Pacman EAST final score**: +1

**Repeat for Pacman actions NORTH and SOUTH**

**5. Final decision**: max(NORTH:+1, EAST:+1, SOUTH:-2) = NORTH or EAST (tie)

### Key Implementation Details

#### 1. Agent Turn Management
```python
# After each agent acts, move to next agent
nextAgent = agentIndex + 1

# When all agents have acted, return to Pacman and increment depth  
if nextAgent == gameState.getNumAgents():
    nextAgent = 0      # Back to Pacman
    nextDepth = depth + 1  # Start new ply
```

#### 2. Depth Progression
- **Depth increments** only when control returns to Pacman
- One **complete ply** = Pacman moves + Ghost 1 moves + Ghost 2 moves
- **Depth 0**: First round - Pacman, Ghost 1, Ghost 2
- **Depth 1**: Second round - Pacman, Ghost 1, Ghost 2

#### 3. Agent Turn Cycle
The complete cycle for one ply with 2 ghosts:
```
Agent 0 (Pacman) → Agent 1 (Ghost 1) → Agent 2 (Ghost 2) → Agent 0 (Pacman, depth++)
```

#### 3. Base Cases
```python
# Stop recursion when:
# 1. Game ends (win/lose)
# 2. Maximum depth reached  
# 3. No legal actions available
if gameState.isWin() or gameState.isLose() or depth == self.depth:
    return self.evaluationFunction(gameState)
```

The Minimax algorithm guarantees that Pacman will make the best possible decision, assuming all ghosts also play optimally to minimize Pacman's score. With multiple ghosts, the algorithm creates a deeper, more complex tree, but the same principle applies: Pacman maximizes while each ghost minimizes in sequence.