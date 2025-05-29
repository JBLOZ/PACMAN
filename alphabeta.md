# Alpha-Beta Pruning

## Optimizing Minimax with Alpha-Beta Pruning

### Introduction

While the Minimax algorithm guarantees optimal decisions, it has a significant drawback: it explores **every possible branch** of the game tree, which can be computationally expensive. **Alpha-Beta Pruning** is an optimization technique that can significantly reduce the number of nodes evaluated without affecting the final result.

### What is Alpha-Beta Pruning?

Alpha-Beta Pruning works by **eliminating branches** that cannot possibly influence the final decision. It maintains two values:

- **Alpha (α)**: The best value that the **maximizer** (Pacman) has found so far
- **Beta (β)**: The best value that the **minimizer** (Ghost) has found so far

### The Pruning Principle

The key insight is:
> If at any point we discover that the current path will not be chosen, we can stop exploring it.

**Specifically:**
- **In MAX nodes**: If the value becomes ≥ β, prune (the minimizer won't allow this path)
- **In MIN nodes**: If the value becomes ≤ α, prune (the maximizer won't choose this path)

### Alpha-Beta Algorithm

```python
def alphabeta(node, depth, alpha, beta, is_maximizing_player):
    if depth == 0 or node.is_terminal():
        return node.evaluate()
    
    if is_maximizing_player:
        max_eval = float('-inf')
        for child in node.get_children():
            eval_score = alphabeta(child, depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:  # Prune!
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.get_children():
            eval_score = alphabeta(child, depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:  # Prune!
                break
        return min_eval
```

## Application to Pacman

Now we will apply Alpha-Beta pruning to the Pacman game, where:
- **Pacman (Agent 0)**: Maximizing player - wants to maximize score
- **Ghosts (Agents 1+)**: Minimizing players - want to minimize Pacman's score

### Game Structure in Pacman

In Pacman Alpha-Beta with **2 ghosts**:
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

#### The Alpha-Beta Game Tree (Depth 2)

```
                    ROOT: Pacman Turn (MAX)
                   α=-∞, β=+∞
                   /          |          \
               NORTH        EAST        SOUTH
              Score: ?    Score: ?    Score: ?
                |            |            |
        
    DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)   DEPTH 0: Ghost1 Turn (MIN)
           α=-∞, β=+∞               α=+1, β=+∞               α=+1, β=+∞
         /    |    \                  /    |    \                  /    ✗    ✗
      N     E     S                N     E     S                N    PRUNED
      |     |     |                |     |     |                |         
      
    DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)   DEPTH 0: Ghost2 Turn (MIN)
           α=-∞, β=+∞               α=+1, β=+∞               α=+1, β=+∞
         /    |    \                  /    |    ✗                |
      N     E     S                N     E   PRUNED            N → returns -1
      |     |     |                |     |                    ↓
      |     |     |                |     |                  PRUNE!
      
    DEPTH 1: Pacman Turn (MAX)   DEPTH 1: Pacman Turn (MAX)   ALL REMAINING
           α=-∞, β=+1               α=+1, β=+1               BRANCHES
    /    |    \                  /    |    \                PRUNED
   N    E    S                  N    E    S                
   |    |    |                  |    |    |                
   
   [... Ghost1 → Ghost2 → Evaluate ...]     
   
 DEPTH 2: EVALUATE            DEPTH 2: EVALUATE      
   +1  +2  +3                   +1   +1  +1          
       ↓                           ↓
   Returns +1                   Returns +1
```

Legend:
- ✗ = Pruned branches (not evaluated)
- α, β = Alpha-beta values at each node
- Numbers = Final evaluation scores

### Step-by-Step Alpha-Beta Execution

#### Initial State
- **Root**: α = -∞, β = +∞
- Pacman tries actions in order: NORTH, EAST, SOUTH

#### Branch 1: Pacman NORTH

**1.1. Pacman moves NORTH**
- Call `alphabeta(1, 0, -∞, +∞, False)` (Ghost 1's turn)

**1.2. Ghost 1's turn (MIN node)**
- α = -∞, β = +∞
- Ghost 1 tries NORTH first → Through complete evaluation → returns +3
- β = min(+∞, +3) = +3
- Ghost 1 tries EAST → returns +2
- β = min(+3, +2) = +2
- Ghost 1 tries SOUTH → returns +1
- β = min(+2, +1) = +1

**1.3. Ghost 1 returns +1**
- Back to root: α = max(-∞, +1) = **+1**

#### Branch 2: Pacman EAST  

**2.1. Pacman moves EAST**
- Call `alphabeta(1, 0, +1, +∞, False)` (Ghost 1's turn)
- **Note**: α is now +1 from previous branch!

**2.2. Ghost 1's turn (MIN node)**
- α = +1, β = +∞
- Ghost 1 tries NORTH → through Ghost 2 → returns +2
- β = min(+∞, +2) = +2

**2.3. Ghost 1 tries EAST**
- Call `alphabeta(2, 0, +1, +2, False)` (Ghost 2's turn)

**2.4. Ghost 2's turn (MIN node)**
- α = +1, β = +2
- Ghost 2 tries NORTH → through complete depth evaluation → returns +1
- β = min(+2, +1) = +1
- Ghost 2 tries EAST → returns +1
- β = min(+1, +1) = +1
- **Check: β(+1) ≤ α(+1)? YES!**
- **PRUNE! Skip Ghost 2 SOUTH**

**2.5. Ghost 2 returns +1**
- Back to Ghost 1: β = min(+2, +1) = +1
- **Check: β(+1) ≤ α(+1)? YES!**
- **PRUNE! Skip Ghost 1 SOUTH**

**2.6. Ghost 1 returns +1**
- Back to root: α = max(+1, +1) = +1

#### Branch 3: Pacman SOUTH

**3.1. Pacman moves SOUTH**
- Call `alphabeta(1, 0, +1, +∞, False)` (Ghost 1's turn)
- **Note**: α is still +1!

**3.2. Ghost 1's turn (MIN node)**
- α = +1, β = +∞
- Ghost 1 tries NORTH → Ghost 2 turn

**3.3. Ghost 2's turn (MIN node)**
- α = +1, β = +∞
- Ghost 2 tries NORTH → Pacman (depth=1) → eventually returns **-1**
- v = min(+∞, -1) = -1
- β = min(+∞, -1) = -1
- **Check: v(-1) < α(+1)? YES!**
- **PRUNE! Return -1 immediately**

**3.4. Ghost 2 returns -1**
- Back to Ghost 1: β = min(+∞, -1) = -1
- **Check: β(-1) ≤ α(+1)? YES!**
- **PRUNE! Skip Ghost 1's EAST and SOUTH**

**3.5. Ghost 1 returns -1**
- Back to root: α remains +1

#### Final Decision

**Root Node Final State:**
- NORTH: +1
- EAST: +1 ← TIE
- SOUTH: -1 (pruned early)

**Pacman chooses NORTH or EAST** (both give +1)

### Implementation for Pacman

```python
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState: GameState):
        """
        Returns the alpha-beta action using self.depth and self.evaluationFunction
        """
        
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: Check if the game is over or if we've reached the maximum depth
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (maximizer) is agentIndex 0
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            # Ghosts (minimizer) are agentIndex 1 or higher
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize max value
            v = float('-inf')
            # Get Pacman's legal actions
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphabeta(1, depth, successor, alpha, beta))  # Ghosts start at index 1
                if v > beta:
                    return v  # Prune the remaining branches
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize min value
            v = float('inf')
            # Get the current agent's legal actions (ghosts)
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)

            # Get the next agent's index and check if we need to increase depth
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  # Go back to Pacman
                depth += 1  # Increase the depth since we've gone through all agents

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphabeta(nextAgent, depth, successor, alpha, beta))
                if v < alpha:
                    return v  # Prune the remaining branches
                beta = min(beta, v)
            return v

        # Pacman (agentIndex 0) will choose the action with the best alpha-beta score
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)  # Start with Ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)

        return bestAction
```

### Complete Execution Trace

Let's trace the alpha-beta execution step by step with 2 ghosts:

**Initial Call**: `getAction(gameState)`

**1. Try Pacman action: NORTH**
   - `generateSuccessor(0, NORTH)` → New state with Pacman at (1,2)
   - Call `alphabeta(1, 0, successor_state, -∞, +∞)`

**2. Ghost 1's turn (agentIndex=1, depth=0)**
   - `minValue(1, 0, state, -∞, +∞)` called
   - Ghost 1 legal actions: [NORTH, EAST, SOUTH]
   
   **2.1 Try Ghost 1 action: NORTH**
   - `generateSuccessor(1, NORTH)` → Ghost 1 at (3,0)
   - nextAgent = 2, nextDepth = 0
   - Call `alphabeta(2, 0, successor_state, -∞, +∞)`
   
   **2.1.1 Ghost 2's turn (agentIndex=2, depth=0)**
   - `minValue(2, 0, state, -∞, +∞)` called
   - Ghost 2 legal actions: [NORTH, EAST, SOUTH]
   
   **2.1.1.1 Try Ghost 2 NORTH** → Eventually returns +3
   - v = min(+∞, +3) = +3
   - β = min(+∞, +3) = +3
   
   **2.1.1.2 Try Ghost 2 EAST** → Eventually returns +2
   - v = min(+3, +2) = +2
   - β = min(+3, +2) = +2
   
   **2.1.1.3 Try Ghost 2 SOUTH** → Eventually returns +1
   - v = min(+2, +1) = +1
   - β = min(+2, +1) = +1
   
   **2.1.1 Ghost 2 returns +1**
   
   **2.1 Back to Ghost 1 NORTH** → value +1
   - v = min(+∞, +1) = +1
   - β = min(+∞, +1) = +1
   
   **2.2 Try Ghost 1 action: EAST** → Eventually returns +2
   - v = min(+1, +2) = +1
   - β = min(+1, +2) = +1
   
   **2.3 Try Ghost 1 action: SOUTH** → Eventually returns +3
   - v = min(+1, +3) = +1
   - β = min(+1, +3) = +1

**3. Ghost 1 returns minimum**: +1
   - Back to root: α = max(-∞, +1) = +1

**4. Try Pacman action: EAST**
   - `generateSuccessor(0, EAST)` → New state with Pacman at (2,1)
   - Call `alphabeta(1, 0, successor_state, +1, +∞)`

**5. Ghost 1's turn (agentIndex=1, depth=0)**
   - `minValue(1, 0, state, +1, +∞)` called
   - **Note**: α is now +1 from previous branch!
   
   **5.1 Try Ghost 1 action: NORTH** → Through Ghost 2 → returns +2
   - v = min(+∞, +2) = +2
   - β = min(+∞, +2) = +2
   
   **5.2 Try Ghost 1 action: EAST** → Through Ghost 2 → returns +1
   - v = min(+2, +1) = +1
   - β = min(+2, +1) = +1
   - **Check: β(+1) ≤ α(+1)? YES!**
   - **PRUNE! Skip Ghost 1 SOUTH action**

**6. Ghost 1 returns +1**
   - Back to root: α = max(+1, +1) = +1

**7. Try Pacman action: SOUTH**
   - `generateSuccessor(0, SOUTH)` → New state with Pacman at (1,0)
   - Call `alphabeta(1, 0, successor_state, +1, +∞)`

**8. Ghost 1's turn (agentIndex=1, depth=0)**
   - `minValue(1, 0, state, +1, +∞)` called
   
   **8.1 Try Ghost 1 action: NORTH**
   - `generateSuccessor(1, NORTH)` → Ghost 1 at (3,0)
   - Call `alphabeta(2, 0, successor_state, +1, +∞)`
   
   **8.1.1 Ghost 2's turn (agentIndex=2, depth=0)**
   - `minValue(2, 0, state, +1, +∞)` called
   
   **8.1.1.1 Try Ghost 2 NORTH** → Eventually returns -1
   - v = min(+∞, -1) = -1
   - β = min(+∞, -1) = -1
   - **Check: v(-1) < α(+1)? YES!**
   - **PRUNE! Return -1 immediately**
   
   **8.1 Ghost 2 returns -1**
   - Back to Ghost 1: v = min(+∞, -1) = -1
   - β = min(+∞, -1) = -1
   - **Check: β(-1) ≤ α(+1)? YES!**
   - **PRUNE! Skip Ghost 1's EAST and SOUTH actions**

**9. Ghost 1 returns -1**
   - Back to root: α = max(+1, -1) = +1

**10. Final decision**: max(NORTH:+1, EAST:+1, SOUTH:-1) = NORTH or EAST (tie)

### Key Alpha-Beta Insights

#### 1. Multiple Minimizer Levels
With 2 ghosts, we have two consecutive MIN levels:
- Ghost 1 minimizes over Ghost 2's choices
- Ghost 2 minimizes over Pacman's future choices
- Both can trigger pruning independently

#### 2. Cascading Pruning
When Ghost 2 triggers pruning (β ≤ α), it not only skips Ghost 2's remaining moves but can also cause Ghost 1 to skip its remaining moves.

#### 3. Agent-Specific Alpha-Beta
```python
# In MIN nodes for both ghosts: prune when β ≤ α
if v < alpha:
    return v  # Don't explore remaining actions

# Update beta for current ghost
beta = min(beta, v)
```

#### 4. Order Matters
The order of exploring children affects pruning efficiency:
```python
# Better move ordering → more pruning
# In practice, try "promising" moves first
```

#### 5. Alpha-Beta Conditions
```python
# In MAX nodes: prune when v ≥ β
if v > beta:
    return v

# In MIN nodes: prune when v ≤ α  
if v < alpha:
    return v
```

#### 6. Maintaining Bounds
```python
# MAX node updates alpha
alpha = max(alpha, value)

# MIN node updates beta
beta = min(beta, value)
```

### Comparison: Minimax vs Alpha-Beta

**Minimax**: Evaluates **81 nodes** in the tree (3^4 for all combinations with 2 ghosts)
**Alpha-Beta**: Evaluates **43 nodes** in the tree 
**Savings**: 47% fewer evaluations!

The savings are even more dramatic with the two-ghost scenario. Notice how:
1. **Early pruning in Ghost 1 EAST branch**: When β ≤ α, we skip Ghost 1's remaining moves
2. **Massive pruning in SOUTH branch**: Both ghosts get pruned early due to the improved α value

### Important Note

**Alpha-Beta pruning returns the exact same result** as regular Minimax - it's just faster! The pruned branches are guaranteed to not affect the optimal decision. With multiple ghosts, the algorithm creates more opportunities for pruning, especially when the first few evaluations establish good alpha and beta bounds that help eliminate large portions of the search tree.

The Minimax algorithm with Alpha-Beta pruning guarantees that Pacman will make the best possible decision, assuming all ghosts also play optimally to minimize Pacman's score, but does so with significantly improved computational efficiency.