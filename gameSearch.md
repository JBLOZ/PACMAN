# Game Search 
## Pacman
**Motivation**. The classical Pacman game illustrates how game playing algorithms work. Basically, <span style="color:#f88146">a game is a combat between our **agent** and another one (or more) agents</span>. The **rules** of this combat are as following (see Chapter 8 in Pearl's book: <span style="color:#f88146">"Heuristics: Intelligent Search Strategies for Problem Solving"</span>):  

1) In principle, there are two **adversarial players** who alternate in turn, each viewing in the failure of the opponent its own success. 
2) The players have **perfect information**. This means that the rules are known for each player and *there is no room for chance*. Each player has complete information about its opponent's position and the choices available to it.

3) The game starts at an **initial state** and ends at a state where using a simple criterion can be declared a WIN, a LOSS or a DRAW. 

In Pacman (see {numref}`Pacman`) the yellow agent is completely aware of the positions of the food and the ghosts at any time. It is also aware of the state of the ghosts (their color moves  to white when they become vulnerable). The game ends either when the yellow agent (say YA) eats all the food (WIN) or it is caught by any of the ghosts (LOSS). There is no DRAW in Pacman but there is in other strategy games such as Chess. 

```{figure} ./images/Topic4/Pacman.png
---
name: Pacman
width: 600px
align: center
height: 400px
---
Pacman agent playing against two ghosts. Source: [The Berkeley Pac-Man Projects](https://inst.eecs.berkeley.edu/~cs188/fa24/projects/)
```

## Minimax Adversarial Search
### Game Trees
A **game tree** is a state-based representation of all possible plays of the game (see {numref}`Gametree`):
1) The **root** node is the <span style="color:#f88146">initial position</span> of the game. 
2) The **successors** of the root are the positions that the <span style="color:#f88146">first player</span> can reach in *one movement*. 
3) The **respective successors** are the positions resulting from the <span style="color:#f88146">second player's replies</span> and so on. First and second player levels alternate. 
4) The **leaves** or terminal nodes of the tree are <span style="color:#f88146">game ending</span> with WIN, LOSS or DRAW. 
5) Any **path** from the root to a leaf represent a <span style="color:#f88146">different complete play</span> of the game. 

 

```{figure} ./images/Topic4/Gametree-removebg-preview.png
---
name: Gametree
width: 800px
align: center
height: 500px
---
Min-Max game tree with two adversarial players. Source: Pearl's book.
```

The two adversarial players are named MAX and MIN. MAX players are represented by squares $\square$, and MIN ones with circles $\bigcirc$. 

The tree in {numref}`Gametree` shows that MAX can win the play even if MIN do its best. Why? Well, to start with, initially the labels L, W and D are only available at the leaves. They climb to the root as follows: 

- **MIN nodes** seek making the MAX player LOOSE. As a result, they will choose from the successors the worst game end wrt MAX. For instance, if the successors of a MIN node are L and W, MIN will choose L. If the successors of the MIN node are L and D, MIN will choose L. Between D and W, MIN will chose D. Finally, <span style="color:#f88146">MIN will choose W if all its successors are W</span>. 

- **MAX nodes** seek making the MAX player WIN. As a result, the will choose from the successors the best game end wrt MAX. Summarizing, <span style="color:#f88146">MAX will choose L if all its successors are L</span>.

This logic is sumarized by a function called $\text{STATUS}(J)$. 

- If $J$ in a **non terminal MAX** node, then we have: 

$$
\text{STATUS}(J)=
\begin{cases}
    \text{WIN} &\text{if ANY of}\;J's\;\text{successors is a WIN} \\[2ex]
    \text{LOSS} &\text{if ALL}\;J's\;\text{successors is a LOSS} \\[2ex]
    \text{DRAW} &\text{if ANY of}\;J's\;\text{successors is a DRAW and NONE is a WIN} \\[2ex]
\end{cases}
$$

- If $J$ in a **non terminal MIN** node, then we have: 

$$
\text{STATUS}(J)=
\begin{cases}
    \text{WIN} &\text{if ALL}\;J's\;\text{successors are WIN} \\[2ex]
    \text{LOSS} &\text{if ANY of}\;J's\;\text{successors is a LOSS} \\[2ex]
    \text{DRAW} &\text{if ANY of}\;J's\;\text{successors is a DRAW and NONE is a LOSS} \\[2ex]
\end{cases}
$$

The above definition of $\text{STATUS}$ shows that MAX and MIN do have an **inverse AND/OR logic**. Note, for instance, that a MAX node WINs if **ANY** of its successors, whereas the MIN node WINs only in **ALL** its successors WIN.

Interestingly, $\text{STATUS}(J)$ is <span style="color:#f88146">the **best terminal status** MAX can achieve from position $J$ if he plays optimally against a perfect oponent</span> (who also plays optimally).

**Solving the game tree** $T$ means <span style="color:#f88146">labeling the root node with WIN, LOSS or DRAW assuming that both opponents play optimally</span>. 

### Strategies
**What is an strategy?**  Given the tree $T$, 

- A strategy for MAX is a sub-tree $T^+$ which contains **one** successor of each non-terminal MAX node of $T^+$, and **all** successors of every non-terminal MIN node of $T^+$. In {numref}`Gametree`, the MAX strategy is the subtree in bold. 
- A strategy for MIN is a sub-tree $T^-$ which contains **one** successor of each non-terminal MIN node of $T^-$, and **all** successors of every non-terminal MAX node of $T^+$. In {numref}`Gametree`, the MIN strategy is the subtree in dashed lines.

**Not all strategies are winning**. In {numref}`Gametree`, the MAX strategy is a winning one since we can label the root with WIN *even if MIN plays optimally*. In other words, with current the terminal labeling, MIN cannot stop MAX winning, even playing its best strategy which is in dashed lines. 

Of course, the success of a given strategy depends on the labeling of the terminal nodes. If all of them are labeled as LOSS (WIN), there is no chance of MAX (MIN) to win, but this is, by definition, contrary to the spirit of a game. 

However, {numref}`Gametree`, shows an interesting fact concerning strategies. If you take the terminal nodes belonging to $T^+$ and $T^-$, respectively, it is easy to check that there is a **single common terminal node**. In addition, the label of this terminal node determines the result of the game when both players adhere to their optimal strategies!

If we call $(T^+,T^-)$ to this **unique** common node, and $s$ is the root, we have: 

$$
\text{STATUS}(s)=\max_{T^+}\min_{T^-}\text{STATUS}(T^+,T^-)=\min_{T^-}\max_{T^+}\text{STATUS}(T^+,T^-)\;.
$$

What's the meaning of this? Well, if MAX chooses $T^+$, MIN can only strike back by choosing the least favorable leaf in $T^+$ which has label $\min_{T^-}\text{STATUS}(T^+,T^-)$. But note that in {numref}`Gametree`. all these leaves are W (this is partially why $T^+$ encodes a winning strategy for MAX). Then if MIN performs $\min_{T^-}\text{STATUS}(T^+,T^-)$ if will obtain W! The case of MIN is symmetric. 

In other words, <span style="color:#f88146">if you discover a winning stategy **it is pointless to reveal it or not** to your opponent (see the case of "checkmate in $x$ steps" in Chess)</span>. Computing $\text{STATUS}(s)$ is more than finding a winning strategy for MAX, it is an **optimization procedure** for games. 

<br></br>
<span style="color:#d94f0b"> 
**Exercise**. Given the tree in {numref}`Gametree`, replace the current values in the leaves to **a)** force a DRAW, and **b)** force a LOSS. Find, in both cases the common node $(T^+,T^-)$. 
<br></br> 
Answer:
<br></br> 
**a)** Let us name the current values of $T$'s leaves as they are found by a depth-first search DFS algorithm (the usual way it is done): top-down and left-to-right. They are (labeled as $t_{ijk}$ where $i$ is the number of child of the MAX root, $j$ is the level, starting at level $0$ for the root, and $k$ is the child number): 
<br></br>
<span style="color:#d94f0b">
$
\begin{align}
\text{1st subtree:}\;\; & t_{121}=W,\;\; t_{131}=D,\;\; t_{141}=L,\;\; t_{142}=D\\
\text{2nd subtree:}\;\; & t_{231}=W,\;\; t_{241}=L,\;\; t_{242}=D,\;\; t_{243}=W,\;\;t_{244}=W,\;\; t_{234}=L\\
\text{3rd subtree:}\;\; & t_{341}=L,\;\; t_{342}=W,\;\; t_{332}=D,\;\; t_{322}=L\\
\end{align} 
$
</span>
<br></br>
<span style="color:#d94f0b">
In order to force a draw, we need to place either DRAW or LOSS in ALL the three subtrees. There is yet a D in the first subree, and there is a L in the third subtree. Then, we only need to force either a D or a L in the second (central). In order to do so, set  $t_{231}=L$ or $t_{231}=D$ instead of its current value $W$. Actually, this is the leaf node corresponding to $(T^+,T^-)$. 
</span>
<br></br>
<span style="color:#d94f0b">
**b)** In order to force a LOSS, we must force a L in ALL the subtrees. A simple way of doing it is to make $t_{121}=L$ in the first subtree and set $t_{231}=L$, the $(T^+,T^-)$ leaf, belonging to the second subtree. 
</span>

### The Minimax Method
**Partial expansions**. Expanding the complete game tree is pointless in most of the games due to the combinatorial explosion. Chess requires $10^{120}$ nodes (takes $10^{101}$ centuries by generating 3 billion nodes per second). As a result, we have to <span style="color:#f88146">rely on partial expansions (up to a given number of levels) and then apply an **evaluation function** $e(J)$ to the leaves $J$ of this partial expansion</span>.

**Evaluation functions**. These are *static scores* evaluating how good is the state of the game wrt MAX. For instance, in Pacman MAX must maximize its distance (Manhattan) wrt the ghosts while minimizing the distance wrt the food. This is encoded by something like 

$$
e(J) = \frac{\text{ghost_distance}(J)}{\text{food_distance}(J) + 1}\;,
$$

where $\text{ghost_distance}(J))$ is the closest Manhattan distance to a ghost, and $\text{food_distance}(J)$ is the closest Manhattan distance to a food point. 

Actually, the default evaluation function of Berkeley's Pacman is as follows: 

$$
e(J) = w_1 \cdot\text{score}(J) + w_2\cdot \text{food_distance}(J) + w_3\cdot \text{capsules_distance}(J)\\ 
+ w_4\cdot\text{ghost_distance}(J) + w_5\cdot\text{scared_ghost_distance}(J)\;,
$$

where: 
- $w_1,\ldots, w_5$ are weights (some of them are negative)
- $\text{score}(J)$ the the positional score due to eating food. 
- $\text{capsules_distance}(J)$ is the closest distance to a *magic capsule* (to make the ghosts edible).
- $\text{scared_ghost_distance}(J)$ closest distance to *scared ghosts* when applicable. 

In addition, there are cases where get WIN (clear the board) or LOSS (captured)! 

**The Minimax Rule**. Expand the MAX-MIN tree up to a number of levels and propagate up the $e(J)$ values of the leaves $J$ as follows: 

1. If $J$ is a leaf, then return $V(J)=e(J)$. 
2. If $J$ is a **MAX** node, return its value $V(J)$ which is equal to the **maximum value** of any of its successors. 
3. If $J$ is a **MIN** node, return its value $V(J)$ which is equal to the **minimum value** of any of its successors. 

The Minimax rule is implemented by the following recursive algorithm: 

```{prf:algorithm} MINIMAX
:label: Minimax

**Input** Root MAX node $J\leftarrow \mathbf{s}$\
**Output** Optimal Minimax Value $V(J)$.

1. **if** $J$ is terminal **then** **return** $V(J)=e(J)$. 
2. **for** $k=1,2,\ldots, b$:
    1. Generate $J_k$, the $k-$th successor of $J$ 
    2. Evaluate $V(J_k)$ $\textbf{[recursive call]}$
    3. **if** $k=1$, set $CV(J)\leftarrow V(J_1)$
    4. **else**: 
        1. **if** $J$ is MAX **then** set $CV(J)\leftarrow \max[CV(J),V(J_k)]$
        2. **if** $J$ is MIN **then** set $CV(J)\leftarrow \min[CV(J),V(J_k)]$
3. **return** $V(J)=CV(J)$
```

Note that
- We call it by calling to $V(\mathbf{s})$ where $\mathbf{s}$ is a MAX node. This triggers a call for each of its successors, which are MIN nodes, until a leaf is reached. The first **leftmost** leaf (step 2.1.1) provides the first clue of the Minimax value of its parent. 
- Each **internal** (not-leave) node updates $CV$ (current value) left-to-right (i.e. in **inorder** DFS) as the **for** loop advances. Once we have $CV(J_1)$ we get $V(J_2)$ and update $CV(J)$ if its proceeds. 
- Curiously, for Pacman we have a MAX level followed by as many MIN levels as ghosts. Then another MAX level occurs, and so on. 

**Example**. In {numref}`Minimax-alfabeta` we show a basic example with $d=4$ levels (we assume that $d=0$ is the root). 

```{figure} ./images/Topic4/Minimax-to-cut-removebg-preview.png
---
name: Minimax-alfabeta
width: 800px
align: center
height: 400px
---
Example of Minimax result with nodes than can been cut. Source: Pearl's book.
```

**From Minimax to $\alpha\beta$**. The main limitation of the MINIMAX algorithm is that, in principle we have to span the **full subtree** until level $d$ each time it is our turn. In {numref}`Minimax-alfabeta`, we have that the **branching factor** is $b=2$. This implies reaching to $b^d$ leaves by expanding $b^0 + b^1 + \ldots + b^{d-1}$ internal nodes. 

However, for some arrangements of the evaluation function on the leaves, <span style="color:#f88146">we may **skip nodes to expand** by exploiting the assumption that MAX/MIN plays the best it can assuming that the adversary does the same</span>.

**Cutoffs**. Consider for instance the MIN node $A$. His leftfost child MAX provides $10$. Then, as soon as the second one node $B$ a value equal or greater than $10$ ($14$ in this case), we don't need to expand the subtree rooted in the $C$. This is an *interesting case of cutoff* that saves many nodes to expand. 

Similarly, once the root receives the value $10$ from the first child, and proceeds to expand the rightmost subtree, the root will stop such expansion as soon as thid rightmost subtree returs a value smaller or equal than $10$ ($5$ in this case). This is another kind of cutoff which is symmetric wrt previous one. 



##  Rubik's cube
**Motivation**. Heuristic search was born to address combinatorial problems in terms of <span style="color:#f88146">state-space expansion</span>. This is clearly exemplified by the well known Rubik's cube, with $3\times 3$ colored stickers per face. Then, given the **solved** cube (left), someone **scrambles** it (right) by applying a sequence of rotations. We have six types of $90^{\circ}$ clockwise rotations or <span style="color:#f88146">states</span> following the [standard notation](https://ruwix.com/the-rubiks-cube/notation/): 

| Clockwise | Rotational Description|
|---|---|
|U  | Upper-horizontal block (top-to-left) |
|D  | Down-horizontal block (front-to-right)|
|R  | Right-vertical block (front-to-top)|
|L  | Left-vertical block (front-to-down) |
|F  | Front-face block (front-to-right) |
|B  | Back-face block (back-to-left)|

| Solved Cube | Scrambled Cube|
|---|---|
| <img src="_images/RubikSol-removebg-preview.png" alt="Descripción de la imagen" width="400" height="400"> |  <img src="_images/Rubik-Scramble-removebg-preview2.png" alt="Descripción de la imagen" width="400" height="400">|

**Permutations**. Note that we have 3 types of move: horizontal, vertical and face (side) twist. In addition, if we rotate the upper block to the left, the opposite is to make the same rotation but **counter-clockwise**. Thus, we have six more rotations: U', D', R', L', F' and B'. Thus, strictly speaking, we have $12$ possible rotations to choose from any configuration. In the state-space language, <span style="color:#f88146">each state **may span** $n=12$ states</span>. 

Mathematically each configuration can be considered as a sequence of $6\times 3\times 3 = 54$ elements containing numbers from $1$ to $6$. <span style="color:#f88146">*How many permutations (with repetition) do we have?*</span> Well, for $n=6$ and $r=54$ we have $n^r\approx 10^{42}$. However, the [symmetries of the problem](https://web.mit.edu/sp.268/www/rubik.pdf) lead to the following reasoning: 
- **Corners**. There are $8$ corners in the cube. Then, we have $n_{corners}=8!$ corner arrangements.
- **Corners Orientations**. Each corner arrangement may have $3$ possible orientations (there are $3$ colors that can face up). Then, we have $n_{CO}=3^{8}$ possibilities per corner permutation. 
- **Edges**. There are $(9-4=5-1=4)\times 3$ non-corner and non-center pieces, called edges. These edges can be arranged in $n_{edges}=12!$ ways. 
- **Edges Orientations**. Since each edge may have $2$ orientations (colors), we have $n_{EO}=2^{12}$. 

The product rule of combinatorics leads to: 

$$
n_{corners}\cdot n_{CO}\cdot n_{edges}\cdot n_{EO} = 8!\cdot 3^{8}\cdot 12!\cdot 2^{12}
$$

Analyzing a bit more the cube, we have that: 
- Only $1/3$ of the permutations have the correct orientations. 
- Only $1/2$ of the permutations have the same edge orientations.
- Only $1/2$ of the latter permutations have the correct "parity" (a concept of **group theory**)

Then, we have

$$
\frac{n_{corners}\cdot n_{CO}\cdot n_{edges}\cdot n_{EO}}{3\cdot 2\cdot 3} = 4.3253\cdot 10^{19}\;\text{moves}\;.
$$

**God's number N**. Given a scrambled cube, *<span style="color:#f88146">What is the minimum number of steps to get back to the initial state?*</span> Well, remember that we can perform $n=12$ moves each time. However, given the first $12$, next time we can only do $11$ (the other one undoes the first move). Then: 

$$
12\cdot 11^{\mathbf{N}-1}\ge 4.3253\cdot 10^{19}\Rightarrow \mathbf{N} \ge 19\;.
$$

Actually, in 2013 [Rokicki et al.](https://tomas.rokicki.com/rubik20.pdf) proved that the "diameter of the Rubik's Cube is $\mathbf{N}=20$", i.e. it can be solved in $20$ moves or less. 

The sequence found for the above case is: 

R' D L' F R F' D' R' D' F' U L' U R R U R' F D R' D' F D' L F R F' D F R' F' L' R

and it has $33$ moves. How is it done? Answering this question leads to study the rudiments of **heuristic search** and a **particular approach** (called Iterative Deepening Search) for the Rubik's Cube (RC in the following). 


## Heuristic Search
### Search Tree 
Classical textbooks such as Pearl's one <span style="color:#f88146">"Heuristics: Intelligent Search Strategies for Problem Solving"</span>, address Heuristic Search (HS) in terms of *<span style="color:#f88146">expanding a search tree</span> from the <span style="color:#f88146">root</span> until the <span style="color:#f88146">target</span> is found*.
1) **Root**. <ins>Initial state</ins> $\mathbf{n}_0$. For the RC, it can be any of the $4.3253\cdot 10^{19}$ configurations corresponding to a scrambled RC.
2) **Target**. <ins>Final state</ins> $\mathbf{n}_F$. Obviously, for the RC we have the configuration where all the faces have uniform color. 
3) **Expansion**. Going from $\mathbf{n}_0$ to $\mathbf{n}_F$ is implemented by <ins>deploying a search tree</ins>, i.e. a graph. Such a deploying is performed acording o a given **search strategy**. One strategy is considered to be <ins>more intelligent than another</ins> if it founds the target minimizing the number of intermediate nodes $\mathbf{n}$ explored.

The well-known $A^{\ast}$ algorithm has the following features:
1) It holds a **search border** (the list OPEN) and a list of **interior nodes** (CLOSED).
2) Aims to minimize a **cost function** $f(\mathbf{n})$ for $\mathbf{n}\in \Omega$ (the search space). The cost function is additive and accounts both for the **best cost** from $\mathbf{n}_0$ to $\mathbf{n}$ as well as **approximated cost** from $\mathbf{n}$ to $\mathbf{n}_F$. 


$$
f(\mathbf{n}) = g(\mathbf{n}) + h(\mathbf{n})\;,
$$

where:
- $g(\mathbf{n})$ is the <span style="color:#f88146">cost of the current path</span> $\Gamma_{\mathbf{n}_0,\mathbf{n}}$ from $\mathbf{n}_0$ to $\mathbf{n}$. Note that the graph is a tree, i.e. we only hold backpointers encoding the path with minimal cost from $\mathbf{n}_0$ to $\mathbf{n}$. Obviously, it satisfies $g(\mathbf{n}_0)=0$. 
- $h(\mathbf{n})$ is an <span style="color:#f88146">**estimation** of the cost</span> from $\mathbf{n}$ to $\mathbf{n}_F$. This function is the <span style="color:#f88146">**heuristic**</span> and it satisfies $h(\mathbf{n}_F)=0$.

```{prf:algorithm} General $A^{\ast}$
:label: Astar

**Inputs** Start node $\mathbf{n}_0$\
**Output** Target node $\mathbf{n}_F$ and path $\Gamma_{\mathbf{n}_0,\mathbf{n}_F}$ or FAILURE.

1. $\text{OPEN}\leftarrow \{\mathbf{n}_0\}$. 
2. **while** $\text{OPEN}\neq\emptyset$:
    1. $\mathbf{n}\leftarrow \arg\min_{\mathbf{n}\in \text{OPEN}}f(\mathbf{n})$
    2. $\text{OPEN}\leftarrow \text{OPEN}-\{\mathbf{n}\}$ // Remove from OPEN and put in CLOSE
    3. $\text{CLOSE}\leftarrow \text{CLOSE}\cup\{\mathbf{n}\}$ 
    4. **if** $\mathbf{n}=\mathbf{n}_F$ **return** ($\mathbf{n}_F$,     $\Gamma_{\mathbf{n}_0,\mathbf{n}_F}$)
    5. ${\cal N}_{\mathbf{n}},\{\Gamma_{\mathbf{n}_0,\mathbf{n}'\in {\cal N}_{\mathbf{n}}}\}\leftarrow \text{EXPAND}(\mathbf{n})$ // Generate neighbors and backpointers
    6. **for** $\mathbf{n}'\in {\cal N}_{\mathbf{n}}$: 

        1. **if** $\mathbf{n}'\not\in\text{OPEN}$ and $\mathbf{n}'\not\in\text{CLOSED}$:

            $f(\mathbf{n}')=g(\mathbf{n}') + h(\mathbf{n}')$ with $g(\mathbf{n}') = g(\mathbf{n}) + c(\mathbf{n},\mathbf{n}')$

        2. **if** $\mathbf{n}'\in\text{OPEN}$ or $\mathbf{n}'\in\text{CLOSED}$:
       
            $\Gamma_{\mathbf{n}_0,\mathbf{n}'}\leftarrow \text{REDIRECT}(\Gamma_{\mathbf{n}_0,\mathbf{n}'})$

        3. **if** $\mathbf{n}'\in\text{CLOSED}$ and Reditect($\mathbf{n}'$)=True:

            1. $\text{CLOSE}\leftarrow \text{CLOSE}-\{\mathbf{n}'\}$ // Reopen
            2. $\text{OPEN}\leftarrow \text{OPEN}\cup\{\mathbf{n}'\}$
            
        
4. **return** FAILURE
```

In summary, $A^{\ast}$ proceeds as follows:
1) It selects the best node $\mathbf{n}$ wrt $f(.)$ in the border to **expand** it. Only when it is select it is moved to CLOSE, not when it is expanded!
2) Expanding a node $\mathbf{n}$ means to **create a neighborhood** of states ${\cal N}_{\mathbf{u}}$.
3) The algorithm **attends** $\mathbf{n}\in{\cal N}_{\mathbf{u}}$ to determine whether $f(.)$ is needed and the backpointers that hold the minimal-cost path $\Gamma_{\mathbf{n}_0,\mathbf{n}'}$ must be adjusted. Excepcionally we way re-open a node. 
4) The algorithm ends either when we find the target $\mathbf{n}_F$ or OPEN is empty. 

As example of application of $A^{\ast}$ to the 8-puzzle problem (see next section) we have {numref}`8-puzzle-Man` 

```{figure} ./images/Topic3/8-puzzle-tree-Man-removebg-preview.png
---
name: 8-puzzle-Man
width: 500px
align: center
height: 600px
---
8-puzzle with Manhattan: Nodes: $271$, Expanded: $164$ 
```

where almost $300$ nodes are expanded, i.e. selected according to Step 2.1. The backpointers from $\mathbf{n}_F$ to $\mathbf{n}_0$ are shown in red and the intensity of the node is the value of the heuristic $h(\mathbf{n})$ (the larger the higher) as we see in the next section. 

### Heuristics
Let us give some details about how to build a basic heuristic. 

**8-Puzzle**. A well-known simplification of jigsaw puzzle problems consists of defining a state $\mathbf{n}$ as an $3\times 3$ matrix of tiles $1\ldots 8$ plus a 'space' named as $0$. Given an initial permutation $\mathbf{n}_0\in\Pi_{8\cup 0}$, the objective is to reach a final permutation $\mathbf{n}_f\in\Pi_{8\cup 0}$ <span style="color:#f88146">by **moving the space**: 'up', 'down', 'left' or 'right'*</span>. 

As we see in {numref}`8-puzzle-show`, moving the space is equivalent to a more human-mind way of moving one of the up to $4$ adjacent tiles to fill the space. In the example, we move from the current-state from the next-state by moving the $0$ left, i.e. by moving the $5$ right. 

```{figure} ./images/Topic3/8puzzle-show-removebg-preview.png
---
name: 8-puzzle-show
width: 800px
align: center
height: 300px
---
8-puzzle: States showing the Manhattan geometry of moves. 
```

Note that *diagonals are not allowed*. This is consistent with the [Taxicab geometry or Manhattan World](https://en.wikipedia.org/wiki/Taxicab_geometry). This provides a **natural heuristic** $h_{Manhattan}(.)$ for estimating the cost from the current state to the target: 

$$
h_{Manhattan}(\mathbf{n})=\sum_{i>0}\underbrace{|\mathbf{n}(i,x)-\mathbf{n}_F(i,x)|}_{\text{x-diff}(i)} + \underbrace{|\mathbf{n}(i,y)-\mathbf{n}_F(i,y)|}_{\text{y-diff}(i)}
$$

where $\mathbf{n}(i,x)$ and $\mathbf{n}(i,y)$ are the $x$ **(col)** and $y$ **(row)**  coordinates of the $i-$th tile (without the space). In the example we have 

| Tile (current-state) | $\mathbf{n}$ coords | $\text{x-diff} + \text{y-diff}$| Cumulative $h_{Manhattan}$|
|---|---|---|---|
| 1 | (2,2)  |abs(2 - 0) + abs(2 - 0) = 4 | 4 |
| 2 | (2,1)  |abs(2 - 0) + abs(1 - 1) = 2 | 6 |
| 3 | (2,0)  |abs(2 - 0) + abs(0 - 2) = 4 | 12 |
| 4 | (1,2)  |abs(1 - 1) + abs(2 - 0) = 2 | 14 |
| 5 | (1,1)  |abs(1 - 1) + abs(1 - 1) = 0 | 14 |
| 6 | (0,0)  |abs(0 - 1) + abs(0 - 2) = 3 | 17 |
| 7 | (0,2)  |abs(0 - 2) + abs(2 - 0) = 4 | 21 |
| 8 | (0,1)  |abs(0 - 2) + abs(1 - 1) = 2 | 23 |
|   |        |                            | 23 |

Obviously $\max h_{Manhattan}(\mathbf{n}) = 8\times 4 = 32$, since $4$ is the largest shortest path in the Taxicab geometry between a tile and its ideal position. Any coincidence in row or column with the ideal position reduces the global cost. For instance, note that tile $5$ is correctly posed and its contribution is $0$. 

Note that computing $h_{Manhattan}$ for the new state $\mathbf{n}'\in {\cal N}_{\mathbf{n}}$ (expanded from $\mathbf{n}$) is quite **incremental**. If $j$ is the 'moved tile' we have 

$$
h_{Manhattan}(\mathbf{n}')=h_{Manhattan}(\mathbf{n}) + \nabla h_{Manhattan}(\mathbf{n}')
$$

where, for the moved tile $j$ we have

$$
\nabla h_{Manhattan}(\mathbf{n}') = \underbrace{|\mathbf{n}'(j,x)-\mathbf{n}_F(j,x)|}_{\text{x-diff}'(j)} + \underbrace{|\mathbf{n}'(j,y)-\mathbf{n}_F(j,y)|}_{\text{y-diff}'(j)} - (\text{x-diff}(j) + \text{y-diff}(j))\;,
$$

which only implies a couple of subractions! 

For the above example, where $h_{Manhattan}(\mathbf{n})=24$, the move of tile $j=5$ to the left (desplacement of the space to the right) makes: 

$$
\begin{align}
\nabla h_{Manhattan}(\mathbf{n}') &= \underbrace{|\mathbf{n}'(5,x)-\mathbf{n}_F(5,x)|}_{\text{x-diff}'(5)} + \underbrace{|\mathbf{n}'(5,y)-\mathbf{n}_F(5,y)|}_{\text{y-diff}'(5)} - (0 + 0)\\
 &= \underbrace{|1-1|}_{\text{x-diff}'(j)} + \underbrace{|0-1|}_{\text{y-diff}'(j)} - (0 + 0)\\
  &= 0 + 1 - (0 + 0)\\
  &= 1\;.
\end{align}
$$

As a result, since $\nabla h_{Manhattan}(\mathbf{n}')>0$ (the **gradient** is positive) we have that $h_{Manhattan}(\mathbf{n}')>h_{Manhattan}(\mathbf{n})$ and the new solution is worse.

Note also that for any $\mathbf{n}$ in the 8-Puzzle, we have that its maximum neighborhood's size is $4$ ($|{\cal N}_{\mathbf{u}}|\le 4$). In the above case, for the current-move (left) only $3$ moves are possible: $j=5$, $j=6$ and $j=3$. Then, we have

| $j$ | New position |Ideal position |$\nabla h_{Manhattan}$|
|---|---|---|---|
| 5 | (1,0) | (1,1)  | 0 + 1 - (0 + 0) = 1 |
| 6 | (1,0) | (1,2)  | abs(1-1) + abs(0-2)- 3 = -1|
| 3 | (1,0) | (0,2)  | abs(1-0) + abs(0-2)- 3 = 0|

which shows that the best local decision is to move $j=6$ down to the space (negative gradient).
<br></br>
<span style="color:#d94f0b"> 
**Exercise**. Given the 'New State' in the above figure (center), 
**a)** Compute the value of $h_{Manhattan}$ for this configuration. **b)** Compute the gradients for all the possible moves. **c)** Identify the best move and update $h_{Manhattan}$ accordingly.
<br></br> 
Answer:
<br></br> 
**a)** $h_{Manhattan}(\mathbf{n})=24$ (can be deduced from above). The associated table is 
<br></br>
</span>
<span style="color:#d94f0b">
$
\begin{aligned}
&\begin{array}{|c|c|c|c|c|}
\hline \hline \text{Tile} &  \text{Initial} &  \text{Ideal} & \text{x-diff + y-diff} & \text{Cumulative}  \\
\hline 
0 & (1,1) & (2,2) & 1+1=2 & 2\\
1 & (2,2) & (0,0) & 2+2=4 & 6\\
2 & (2,1) & (0,1) & 2+0=2 & 8\\
3 & (2,0) & (0,2) & 2+2=4 & 12\\
4 & (1,2) & (1,0) & 0+2=2 & 14\\
5 & (1,0) & (1,1) & 0+1=1 & 15\\
6 & (0,0) & (1,2) & 1+2=3 & 18\\
7 & (0,2) & (2,0) & 2+2=4 & 22\\
8 & (0,1) & (2,1) & 2+0=2 & 24\\
\hline
\end{array}
\end{aligned}
$
</span>
<br></br>
<span style="color:#d94f0b">
**b)** Gradients $\nabla h_{Manhattan}(\mathbf{n}')$ for $\mathbf{n}'\in {\cal N}_{\mathbf{n}}$. There are $4$ neighboring tiles with $j=8,5,2,4$ respectively.
</span>
<br></br>
<span style="color:#d94f0b">
$
\begin{aligned}
&\begin{array}{|c|c|c|c|c|}
\hline \hline j &  \text{New} &  \text{Ideal} & \text{x-diff'(j)}+ \text{y-diff'(j)} & \text{x-diff(j)}+ \text{y-diff(j)} & \nabla h_{Manhattan}(\mathbf{n}')\\
\hline 
8 & (1,1) & (2,1) & |1-2|+|1-1|=1 & 2 & 1-2 = -1\\
5 & (1,1) & (1,1) & |1-1|+|1-1|=0 & 1 & 0-1 = -1\\
2 & (1,1) & (0,1) & |1-0|+|1-1|=1 & 2 & 1-2 = -1\\
4 & (1,1) & (1,0) & |1-1|+|1-0|=1 & 2 & 1-2 = -1\\
\hline
\end{array}
\end{aligned}
$
</span>
<br></br>
<span style="color:#d94f0b">
**c)** Best move? The above table shows that all moves are equally good. Why? A human would tend to move $5$ to the center. Such a move would end up in $8$, $5$ and $2$ in the correct column. However, since $8$ and $2$ are in an **inverted position**, the Manhattan heuristic is not able to run in favor of centering $5$. Even moving $4$ to the center and approach it to its ideal column is equally valid! 
</span>

[//]:https://www.tjhsst.edu/~rlatimer/ai2007/Korf-slides.pdf
[//]:https://cse.sc.edu/~mgv/csce580sp15/gradPres/HanssonMayerYung1992.pdf

**Improving Manhattan**. Inversions or <span style="color:#f88146">linear conflicts</span> (two tiles in the correct row or column but in inverted order) are powerful structural violations due to the symmetry of the problem. The <span style="color:#f88146">Manhattan distance is **blind** to linear conflicts</span> because it accounts for the shortest paths (in the Taxicab geometry) of each tile, independently of the others. 

Basically, the following algorithm, proposed by [Hanson et al](https://www.sciencedirect.com/science/article/abs/pii/002002559290070O) in Information Sciences 

```{prf:algorithm} Linear-Conflicts
:label: LC

**Inputs** A 8-puzzle state $\mathbf{n}$\
**Output** $h_{Manhanttan}(\mathbf{n}) + h_{LC}(\mathbf{n})$

1. **for** each row $r_i$ of $\mathbf{n}$:
    1. $lc(r_i,\mathbf{n})\leftarrow 0$
    2. **for** each col $j\in r_i$:
       1. Compute $C(j,r_i)$: the number of LCs with $j$. 
       2. **while** $\exists j: C(j,r_i)>0$: 
         
            1. Remove $k$ with maximal LCs in $r_i$
            2. $C(k,r_i)\leftarrow 0$
            3. **for** each col $j$ which has in conflict with $k$: $C(j,r_i)\leftarrow C(j,r_i)-1$
            4. $lc(r_i,\mathbf{n})\leftarrow lc(r_i,\mathbf{n}) + 1$
2. **repeat** 1 for cols and compute $lc(c_j,\mathbf{n})$
3. $h_{LC}(\mathbf{n})\leftarrow 2(\sum_{i}lc(r_i,\mathbf{n}) + \sum_{j}lc(c_j,\mathbf{n}))$

4. **return** $h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n})$
```

accounts for linear conflicts (LCs) in the following way:
1) Analize row by row. If a LC is found, add $2$ to Manhattan. 
2) Repeat for columns. 

The basic idea of **Linear-Conflicts** is that the Manhattan heuristic is only worried about the independent shortest paths of each tile. In this regard, an inversion is seen as 'doubling' the shortest paths needed (extra effort) because they 'become in conflict'.

For instance, in {numref}`8-puzzle-show` we have the following cases:
1) 'Current State' (left-image). Concerning rows, only $r_1$ has an inversion ($5-4$) and this results in a penalization of $2$. Concerning columns, only $c_1$ has inversion, but we have $2$ conflicts per tile  (e.g. $8-5$ and $5-2$). This results in two iterations of the while loop to make all zeros and the resulting penalization is $4$. Then 

$$
h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n}) = 23 + 2 + 4 = 29\;.
$$

2) 'New State' (center). Regarding the rows, we have a unique conflict ($5-4$) as before (note that the space does not count). Actually, due to the presence of the space in the center, we have only a column conflict. As a result

$$
h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n}) = 24 + 2 + 2 = 28\;.
$$

#### Admissible heuristics
Heuristics $h(\mathbf{n})$ <span style="color:#f88146">**approximate** the true (but unknown)</span> cost $h^{\ast}(\mathbf{n})$ from $\mathbf{n}$ to the target $\mathbf{n}_F$. There are formal reasons recommending 

$$
h(\mathbf{n})\le h^{\ast}(\mathbf{n})\;\;\forall \mathbf{n}\in\Omega, 
$$

which is <span style="color:#f88146">called the **admissibility**</span> of the heuristic. The intuition behind admissibility is that as far as we make 'optimistic' approximations we are sure that $A^{\ast}$ will find the target $\mathbf{n}_F$. Otherwise, i.e. considering a state worse that it really is (being pessimisit), $A^{\ast}$ may skip it and produce a sub-optimal solution if any. 

Some formal considerations (see Pearl's book for more details): 
1) <ins>Admissibility ensures</ins> that at any time before termination, there will be *at least* a node $\mathbf{n}\in\text{OPEN}$ whose expansion will lead to find $\mathbf{n}_F$. 
2) This can be expressed in terms of $f(\mathbf{n})\le C^{\ast}$, where $C^{\ast}$ is the optimal cost from $\mathbf{n}_0$ to $\mathbf{n}_F$ and it is consistent with the <ins>principle of optimality</ins> (all parts of an optimal path are also optimal).

**Is Manhattan admissible?** Yes, it is. But why?
1) Remember that $h_{Manhattan}(\mathbf{n})$ adds the shortest paths from any tile to its ideal position in the Taxicab geometry (no diagonals), while *assuming no obstacles in between*.
2) For a particular tile, it is impossible to make less movements since there are frequently other tiles in between. 

**What about the admissibility of $h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n})$?** Well, this is a bit tricky. 
1)  We know that $h_{Manhattan}(\mathbf{n})$ is admissible. Thus, the above question reduces whether to penalizing LCs as we do is still admissible.
2) The proof is reduced to test whether at each 'line' (row or column) we calculate the <ins>minimum number of tiles which must take non-shortest paths</ins>. 
3) The algorithm removes conflicting tiles and each removal counts $2$ moves, which is the minimal number of moves to solve a LC. 
4) The big question is whether the LCs in a line are independent of those in another. They really are because removing a tile for solving a conflict will not affect the others. If the tile is not in its ideal solution this is obvious. Otherwise, moving it out of this line will not affect possible conflicts in the perpendicular line since we leave a space. 

Thus, $h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n})$ is still admissible!

#### Pruning power 
Let us start by defining the <span style="color:#f88146">**heuristic power**</span> of a given $h(\mathbf{n})$ as the <span style="color:#f88146">*number of nodes expanded* for the same problem instance</span>. 

In {numref}`8-puzzle-Man`, we showed that for $h_{Manhattan}(.)$, $A^{\ast}$ generates $271$ nodes, from which $164$ are expanded. Remember that 'expansion' of $\mathbf{n}$ implies that this node is 'selected' from OPEN according to minimizing $f(\mathbf{n})$. 

However, $h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n})$ leads to $198$ nodes, where $120$ are expanded (see {numref}`8-puzzle-LC`) for the same initial state which is:

$$
\mathbf{n}_0 = [2, 3, 0, 1, 8, 6, 5, 7, 4]
$$

where we linearize the $3\times 3$ puzzle by stacking their rows. 

```{figure} ./images/Topic3/8-puzzle-tree-LC-removebg-preview.png
---
name: 8-puzzle-LC
width: 500px
align: center
height: 600px
---
8-puzzle with LCs: Nodes: $198$, Expanded: $120$ 
```

Intuitively, we see that $h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n})$ is <span style="color:#f88146">**more informed**</span> than the plain $h_{Manhattan}(\mathbf{n})$. More formally, $h_2(.)$ is more informed than $h_1(.)$ if both are admissible and 

$$
h_2(\mathbf{n}) > h_1(\mathbf{n})\;\;\forall \mathbf{n}\in\Omega\;. 
$$

This results in the fact that the number of nodes expanded by $A^{\ast}$ with $h_2(.)$ is **upper-bounded** by those expanded with $h_1(.)$. In other words, the <span style="color:#f88146">**pruning power** of more pessimistic admissible heuristic is larger</span>. 

The rationale exposed in Pearl's book is as follows: 
1) $h^{\ast}(\mathbf{n})$ is a **perfect discriminator**, i.e. it provides the minimal number of expansions. 
2) Since $h_2(\mathbf{n}) \ge h_1(\mathbf{n})$, we have 

$$
h_1(\mathbf{n}) \le h_2(\mathbf{n})\le h^{\ast}(\mathbf{n})\;,
$$

under admissibility.

This is a consequence of <span style="color:#f88146">**Nilsson's theorem**: *Any node expanded by $A^{\ast}$ cannot have an $f$ value exceeding the optimal cost $C^{\ast}$*</span> i.e. 

$$
f(\mathbf{n})\le C^{\ast}\;\;\text{for all expanded nodes}\;.
$$

In other words, *every node on OPEN for which $f(\mathbf{n})<C^{\ast}$ will be eventually expanded by $A^{\ast}$*.

Remember that we cannot select a non-expanded node to determine whether we have found $\mathbf{n}_F$. This node must be a yet expanded node and this means that is satisifies $f(\mathbf{n})\le C^{\ast}$. In other words, the nodes in ${\cal S}=\{\mathbf{n}:f(\mathbf{n})>C^{\ast}\}$ are definitely **excluded from expansion**. This means that better informed heuristics provide better upper bounds and larger **excluded-from-expansion sets** (see a more formal proof in Pearl's page 81). 

**Relaxation**. Since there are configuations where we do not have LCs, what we have is $h_{Manhattan}(\mathbf{n})\le h_{Manhattan}(\mathbf{n}) + h_{LC}(\mathbf{n})$ instead of a $<$. Then, we can relax a bit the requirement of pruning power and admit $\le$. 

**Computational cost**. It seems a kind of obvious that more informed heuristic require more computational cost, for instance, $h_{LC}$ takes $O(N^{1.5})$ whereas $h_{Manhattan}$ takes $O(N)$ where $2\sqrt{N+1}$ is the number of lines (horizontal and vertical) of the puzzle. 

Thus, <span style="color:#f88146">pruning power (spatial complexity) and computational cost (temporal complexity) are tied by a strict trade-off</span>. 'Ask me for memory or for computer power', quoted Steve Jobbs. 


### Failure condition
Remember, that the **failure condition** of $A^{\ast}$ is $\text{OPEN}=\emptyset$, i.e. there are no more nodes to expand and $\mathbf{n}_F$ was not found. This cannot happen *unless* the target cannot be found for any reason. 

[//]:https://math.stackexchange.com/questions/293527/how-to-check-if-a-8-puzzle-is-solvable

**8-Puzzle and even LCs**. Consider the $\mathbf{n}_0$ used in the previous example: 

$$
\mathbf{n}_0 = [2, 3, 0, 1, 8, 6, 5, 7, 4]\;,
$$

reformatted properly as a $3\times 3$ matrix: 

$$
\mathbf{n}_0=\begin{bmatrix}
2 & 3 & 0\\
1 & 8 & 6\\
5 & 7 & 4\\
\end{bmatrix}
$$

has $0$ LCs (herein $0$ is 'even'). 

However the following initial state: 

$$
\mathbf{n}_0=\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6\\
0 & 8 & 7\\
\end{bmatrix}
$$

has $1$ LC (the one given by $8-7$). Well, <span style="color:#f88146">as $1$ is **'odd'**, this 8-puzzle **cannot** be solved!</span> Let us see why. 

[//]:https://puzzling.stackexchange.com/questions/52110/8-puzzle-unsolvable-proof

**Permutations and parity**. Given the sequence $\pi_1=[1,4,3,2,5]$, the canonical order $\pi_0[1,2,3,4,5]$ can be obtained by a single (odd) interchance or <span style="color:#f88146">transposition</span> (simply interchanging $4$ and $2$). However undoing $\pi_2=[4,1,3,2,5]$, **sequentialy** requires two (even) moves: (1) first, interchange $4$ and $1$, thus arriving to $\pi_1=[1,4,3,2,5]$ and then (2) undo $\pi_1$ by interchanging $2$ and $4$ back to the canonical sequence. 

It can be proved that solving an even (odd) number of transpositions requires and even (odd) number of moves. A given permutation can be solved in, say $5$ moves even if it has $3$ transpositions, but what is invariant in the <span style="color:#f88146">**parity** of a permutation (odd or even) is always preserved</span>.

As a result, it is almost trivial to proof that <span style="color:#f88146">8-puzzles have **even** parity</span>. In other words, if the initial state has an odd parity, the puzzle is unsolvable! 

One may think that we can exploit the 'space' to change the parity of the permutations encoded by the 8-puzzle but:
1) Look that horizontal moves do not change the permutation at all. 
2) Only vertical moves change the permutations and we know that the original parity **must** be perserved. 
3) In other words, <span style="color:#f88146">*it is impossible to achieve a state with even parity from the target puzzle backwards!*</span>

Therefore, for avoiding the failure condition is convenient to run a parity check before calling to $A^{\ast}$.

### Pattern Databases
**Computational cost**. It seems a kind of obvious that more informed heuristic require more computational cost, for instance, $h_{LC}$ takes $O(N^{1.5})$ whereas $h_{Manhattan}$ takes $O(N)$ where $2\sqrt{N+1}$ is the number of lines (horizontal and vertical) of the puzzle: $N=8$ in the 8-Puzzle. 

Thus, <span style="color:#f88146">pruning power (spatial complexity) and computational cost (temporal complexity) are tied by a strict trade-off</span>. 'Ask me for memory or for computer power', quoted Steve Jobbs. However, even with this trade-off at hand, the time needed to solve N-Puzzles can be un-practical, since N-Puzzles are NP-hard problems in general due to the their permutational nature. 


[//]:https://web.cs.umass.edu/publication/docs/1987/UM-CS-1987-071.pdf

**Lookup Tables**. Let us try to <ins>pre-compute</ins> the <span style="color:#f88146">perfect discriminator</span> $h^{\ast}(\mathbf{n})$ to keep the number of expanded nodes in $\mathbf{A}^{\ast}$ at a minimum. 

For problems such as 8-Puzzle or Rubik where the <ins>final state is always the same</ins>, lookup tables become a useful tool. They work as follows: 

1) **Start** from the 'target' state and its distance to it (zero). 
2) **Extract** next state to explore from a QUEUE (push/pop FIFO operators). 
3) **Expand** the search tree using <span style="color:#f88146">BFS (Best-First Search)</span> and register the effective distance to the target. 
4) Create an **entry** for each 'new state' is found (not seen before). Store the state and its cost to the target: $T(\mathbf{n},\text{cost})$. 
5) **Stop** when not new state can be found. 

More algorithmically, we have

```{prf:algorithm} Lookup-Table
:label: LU

**Inputs** Goal node $\mathbf{n}_F$\
**Output** Lookup table $T:\Omega\rightarrow \mathbb{N}$

1. $\text{OPEN}\leftarrow \{(\mathbf{n}_F,0)\}$. 
2. $T(\mathbf{n}_F)=0$
3. **while** $\text{OPEN}\neq\emptyset$:
    1. $(\mathbf{n},d)\leftarrow \text{pop}(\text{OPEN})$
    2. ${\cal N}_{\mathbf{n}}\leftarrow \text{EXPAND}(\mathbf{n})$  
    3. **for** $\mathbf{n}'\in {\cal N}_{\mathbf{n}}$: 

        1. **if** $\mathbf{n}'\not\in T$:

            1. $T(\mathbf{n}')=d+1$

            2. $\text{OPEN}\leftarrow \text{push}(\text{OPEN},\mathbf{n}')$
                    
4. **return** $T$
```

Then, since $T(\mathbf{n})=h^{\ast}(\mathbf{n})$, lookup tables allow us to characterize the distribution of optimal distances. In {numref}`8-puzzle-LUT` we show that the 8-puzzle can be solve in $31$ moves. In addition, we see that: 
-  Most of the states have distances between 20-25 moves. 
-  The distribution is asymmetric towards large distances, but it is very far from being equiprobable. 
- If our target state changes (for instance placing the 'space' in the center), we should re-compute the table. 

```{figure} ./images/Topic3/8-puzzle-lookup-removebg-preview.png
---
name: 8-puzzle-LUT
width: 600px
align: center
height: 500px
---
Distribution of distances for the Lookup table of the 8-Puzzle
```

Interestingly, the fact that $T(\mathbf{n})=h^{\ast}(\mathbf{n})$ minimizes the number of expansions. For the same initial space as before we expand only $21$ nodes (see {numref}`8-puzzle-astar-lookup`).


```{figure} ./images/Topic3/8-puzzle-astar-lookup-removebg-preview.png
---
name: 8-puzzle-astar-lookup
width: 500px
align: center
height: 600px
---
8-puzzle with Lookup: Nodes: $39$, Expanded: $21$ 
```

However, note that the lookup table is huge in larger problems (e.g. Rubik's Cube) and cannot be applied to problems where the target state changes (e.g. graph matching, TSPs, etc.)


## Iterative Deepening
Iterative Deepening is the practical approach of $A^{\ast}$-inspired techiques fueled by a huge lookup table. This has been the *de facto* standard for thr Rubik's Cube until very recently! 

### Mixed Strategies  
We should see $A^{\ast}$ as a **mixed strategy** between BFS (Breath-First-Search) and DFS (Deep-First Seatch). Actually: 
- <ins>BFS results from seting $h(\mathbf{n})=0$ (only $g(\mathbf{n})$ counts)</ins>. This result in an innaceptable memory requirement.  
- <ins>DFS results, however, from setting $g(\mathbf{n})=0$ instead (only $h(\mathbf{n})$ counts)</ins>, and expanding the nodes until a given depth cutoff $d$ is reached. This solves the problem of memory requirement but $d$ is generally unknown. 

### DF Iterative Deepening
This is a *brute force* algorithm that suffers neither the drawbacks of BFS nor DFS. It works as follows:
1) Perform DFS for $d=1$.
2) **Discard** the nodes generated in the $d$ search and make a new search for $d=d+1$
3) Do 2) until the target state is found. 

Discarding all the nodes generated for a given $d$ and start again for $d+1$ seems to be very inefficient. However, [Richard E. Korf](https://www.cse.sc.edu/~mgv/csce580f09/gradPres/korf_IDAStar_1985.pdf) proved that this is not the case. The algorithm is **asymptotically** optimal among brute-force tree searches in terms of space, time and the length of the solution. 

The proof is quite ilustrative of how **branching processes** work in practice. 
1) Consider a tree starting at the root $\mathbf{n}_0$ and a constant branching factor $b=|{\cal N}_{\mathbf{n}}|$.
2) Then, the total number of nodes generated at depth $d$ are: 

$$
S_b = b^0 + b^1 + \ldots b^{d} = \sum_{i=0}^d b^d = b\left(\frac{1-b^{d+1}}{1-b}\right)\;,
$$

i.e. the sum of a geometric series with ratio $r = \frac{b^{d+1}}{b^{d}}=b$. For instance, is $b=2$ we have 

$$
S_2=2\frac{(1-2^{d+1})}{-1}=2^d - 1\;.
$$
 
 Well, when DF Iterative Deepening (DFID) is applied we have the following for depth $d$:
 - The root is generated $d$ times. 
 - The first level of successors is generated $d-1$ times. 
 - The $i-$th level of successors is generated $d-i$ times.
 - Level $d$ is generated only once.  
 
 Then the number of nodes generated up to level $d$ is: 

 $$
 (d-0)b^1 + (d-1)b^2 + \ldots (d-i)b^{i+1} + \ldots + 3b^{d-2} + 2b^{d-1}+ b^d
 $$

 Inverting the order we have 

 $$
 b^d + 2b^{d-1} + 3b^{d-2} + \ldots + db\;.
 $$

 Factoring $b^d$ yields

 $$
 b^d(1 + 2b^{-1} + 3b^{-2} + \ldots + db^{1-d})\;,
 $$

 and making $x=b^{-1}$ unveils an interesting series

 $$
 b^d(1 + 2x + 3x^2 + \ldots dx^{d-1})
 $$

which converges for $d\rightarrow\infty$ as follows

$$
b^d(1 + 2x + 3x^2 + \ldots )\rightarrow b^d(1-x)^{-2}\;\;\text{for}\; |x|<1\;. 
$$

Since $(1-x)^{-2}=(1-1/b)^{-2}$ is a constant that is independent of $d$, for $b>1$ we have that the **temporal complexity** of DFID is $O(b^d)$, basically that of BFS. This is because the geometric part of the series dominates the arithmetic part (there are so much nodes generated as depth increases that their number grows faster than their repeats).

Considering now the **space complexity**, since DFID is engaged in a DFS it only stores the nodes of the branch leading to the maximum depth, wich only takes $O(d)$.

Naturally, the **waste factor** is upper-bounded by $(1-1/b)^{-2}$, i.e. <span style="color:#f88146">the largest the branching factor $b$, the smaller is the maximal waste</span>. Taking derivatives, the rate of such decrease is $O(1/b^2)$. 

### ID$A^{\ast}$
[Iterative-Deepening $A^{\ast}$](https://en.wikipedia.org/wiki/Iterative_deepening_A*) results from combining DFID with a BFS such as $A^{\ast}$. The general idea can be summarized as follows:

1) Instead of repeating the search **blindly** as in DFID, ID$A^{\ast}$ expands many deep-first searches from the root $\mathbf{n}_0$ until one of them 'hits' the target $\mathbf{n}_F$ or it discovers that it is much far away. 
2) The depth of each of these searches is controled by a **bound** which starts with $h(\mathbf{n}_0)$ and cuts off a branch ending in $\mathbf{n}$ when $f(\mathbf{n})=g(\mathbf{n})+h(\mathbf{n}) > t$ where $t$ is a threshold. At each iteration, $t$ is the minimum of all the values that exceeded the current threshold (the less agressive excess).

The algorithm is as follows: 

```{prf:algorithm} ID$A^{\ast}$
:label: IDAstar

**Inputs** Root $\mathbf{n}_0$\
**Output** Path $\Gamma$ to target node $\mathbf{n}_F$ and Bound, NOT_FOUND or FAILURE

1. $\text{bound}\leftarrow h(\mathbf{n}_0)$. 
2. $\Gamma\leftarrow [\mathbf{n}_0]$
3. **while** True:
    1. $t \leftarrow \text{BoundedSearch}(\Gamma, 0, \text{bound})$
    2. **if** $t=$FOUND **then** **return** $\Gamma,\text{bound}$
    3. **if** $t=\infty$ **then** **return** NOT_FOUND
    4. $\text{bound}\leftarrow t$
```
Where $\text{BoundedSearch}$ is a recursive bounded DFS guided by $f = g + h$ as follows: 

```{prf:algorithm} $\text{BoundedSearch}$
:label: IDAstar2

**Inputs** Path, $g$ and $\text{bound}$\
**Output** $t$ or FOUND

1. $\mathbf{n}\leftarrow \text{last}(\Gamma)$ 
2. $f\leftarrow g + h(\mathbf{n})$
3. **if** $f>\text{bound}$ **then** **return** $f$ // Returns $f$ as $t$
4. **if** $\mathbf{n}=\mathbf{n}_F$ **then** **return** FOUND 
5. $\text{min}\leftarrow \infty$
6. **for** $\mathbf{n}'\in {\cal N}_{\mathbf{n}}$:
    1. **if** $\mathbf{n}'\not\in\Gamma$:
        1. $\Gamma\leftarrow \text{push}(\Gamma,\mathbf{n}')$

        2. $t \leftarrow \text{BoundedSearch}(\Gamma, g + c(\mathbf{n},\mathbf{n}'), \text{bound})$

        3. **if** $t=$FOUND **then** **return** FOUND

        4. **if** $t<\text{min}$ **then** $\text{min}\leftarrow t$

        6. $\Gamma\leftarrow \text{pop}(\Gamma)$ // Alternative DFS

7. **return** $\text{min}$ 
```

Some considerations: 
1) The search **succeeds** (returns FOUND) as soon as one of the paths expanded by the DFS reaches $\mathbf{n}_F$. 
2) If so, there will be other partial expanded paths, because thei **exceeded** the bound and did not find the target. 
3) If $h$ is **admissible**, ID$A^{\ast}$ always finds a solution of least cost if it exists!

This results in a even more 'skeletal' search (see {numref}`8-puzzle-IDA-lookup`) in comparision with $A^{\ast}$ with lookup table (see {numref}`8-puzzle-astar-lookup`).

```{figure} ./images/Topic3/8-puzzle-IDA-lookup-removebg-preview.png
---
name: 8-puzzle-IDA-lookup
width: 500px
align: center
height: 600px
---
8-puzzle with ID$A^{\ast}$ using Lookup: Nodes: $24$, Expanded: $24$ 
```

### ID$A^{\ast}$ for Rubik
ID$A^{\ast}$ with lookup table has been the standard approach to solve Rubik's Cube (RC) until very recently. Remember that the size of the search space is 

$$
|\Omega| = 43,252,003,274,489,856,000\;,
$$

which requires $128$ GB of memory! If you can access a Google Colab account with a 51 GB limit, we can only account for moves up to length $8$. In {numref}`Rubik-LUT` we show the distances of $8\times 10^6$ moves. Note the exponential increment in the number of nodes with distance!

```{figure} ./images/Topic3/Rubik-Table-Photoroom.png
---
name: Rubik-LUT
width: 600px
align: center
height: 500px
---
(Some) Distribution of distances for the Lookup table of Rubik
```



[Richard E. Korf](https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf) addressed this problem in 1997 by applying ID$A^{\ast}$ making  interesting findings: 
1) The <ins>3D generalization of the Manhattan</ins> distance (number of moves required to correctly position and oriented each 'cubie'), again considering the cubies independent. The sum of the moves of all cubies is divided by $8$ to ensure admissibility. 
2) <ins>A better heuristic</ins> is to take the maximum of the Manhattan distances of the corner cubies (3 orientations each) and the edge cubies (2 orientations each). The expected distance for the edge cubies is $5.5$ whereas that of the corner ones is $3$. 
3) Another solution consists of computing <ins>partial lookup tables storing Manhattan distances</ins> (e.g. for the corner cubies, for the edge cubies, etc) and combine them. 

However, the <ins>above solutions are not enough to contain the above combinatorial explosion</ins> and ID$A^{\ast}$ is only able to compute some movemens per day!

## Learnable Heuristics
[DeepCubeA](https://www.nature.com/articles/s42256-019-0070-z) is the flagship solution of the current **change of paradigm**, where admissibility becomes only a <span style="color:#f88146">**conceptual guide**</span> for solving large problems such as the Rubik's Cube (RC) and it is replaced by:

a)  **Deep Oracles**. In practice, it is assumed that we can only sample the space state $\Omega$. Doing so, an AIer can learn a predictor or <span style="color:#f88146">**oracle**</span> $f_{\theta}(\mathbf{n})=\mathbf{n}'$, so that for any $\mathbf{n}\in\Omega$, then $\mathbf{n}'$ is the closest state to the target state $\mathbf{n}_F$ (e.g the perfect Rubik's solution). <span style="color:#f88146">The oracle can be interpreted as a maximizer of the probability</span> $p_{\theta}(\mathbf{n}'|\mathbf{n})$ of reaching the target from $\mathbf{n}$ via $\mathbf{n}'$. The oracle is learnable, i.e. we must find the parameters $\theta$, via <span style="color:#f88146">**Deep Neural Networks**</span> (DNN). 

b) **Trade-off between computation and optimality**. DNNs <span style="color:#f88146">aim to discover the unknown state space $\Omega$ in order to make **better and better predictions**</span>. Good predictions are those that match **optimal solutions** (e.g. those solving the game), but achieving them requires increasing levels of search. Then, DNNs can achieve <span style="color:#f88146">**near-optimal solutions**, i.e. acceptable solutions with a reasonable amount of time!</span>

### Deep Oracles 
Assume that, given $\mathbf{n}\in\Omega$ the perfect discriminator $h^{\ast}(\mathbf{n})$ cannot be computed but **approximated**. How is such an approximation computed?

Let us explain this process for the Rubik's Cube (RC). 

**Training set**. Given that the target $\mathbf{n}_F$ for RC is fixed and well known, as well as the **overshooted God's Number** $N>20$, let us sample a set of paths $P=\{\Gamma_i\}$ for $i=1,2,\ldots,|P|$. These paths start at $\mathbf{n}_F$ and go backwards by **scrambling** the RC, i.e. the $i-$th path has the following structure: 

$$
\Gamma_{i}=\{(\mathbf{n}_F=\mathbf{n}^{i}_{0})\rightarrow \mathbf{n}^{i}_1\rightarrow\mathbf{n}^{i}_2\rightarrow\ldots \rightarrow\mathbf{n}^{i}_{N}\}
$$

Each $\Gamma_{i}$ <span style="color:#f88146">is a **random walk** through $\Omega$ in reverse order from $\mathbf{n}_F$ backwards using **legal moves** $\mathbf{n}^{i}_{k}\rightarrow \mathbf{n}^{i}_{k+1}\in {\cal M}$</span> and

$$
{\cal M} = \{U,U',D,D',R,R',L,L',F,F',B,B'\}
$$

The above paths *are not exclusive* and may visit a given state several times. There are, however, some optimizations are done in order to avoid 'do-undo' moves or 'three consecutive moves that are really one'.  


**Encoding and Association**. Given $\Gamma_i$, <span style="color:#f88146">only the corresponding $\mathbf{n}_N$ is **observable**</span>. Actually, for the Rubik Cube, $\mathbf{n}^i_N$ is **encoded** by the colors of the $6$ faces, each one having $3\times 3=9$ tiles. Therefore, $\mathbf{n}_N$ can be vectorized with $3\times 3\times 6= 54$ parameters.

During <span style="color:#f88146">**training (offline phase)**</span>, we learn a function $f_{\theta}:\mathbb{R}^{54}\rightarrow [0,1]^{12}$: 

$$
f_{\theta}(\mathbf{n}^i_N)=\left[
    \begin{array}{c}
    p(\mathbf{n}^{i}_{N}\rightarrow \mathbf{n}^{i}_{N-1}=m_1)\\
    p(\mathbf{n}^{i}_{N}\rightarrow \mathbf{n}^{i}_{N-1}=m_2)\\
    \ldots\\
    p(\mathbf{n}^{i}_{N}\rightarrow \mathbf{n}^{i}_{N-1}=m_{12})\\
    \end{array}
    \right] =\left[
     \begin{array}{c}
    p(m_1|\mathbf{n}^{i}_{N})\\
    p(m_2|\mathbf{n}^{i}_{N})\\
    \ldots\\
    p(m_{12}|\mathbf{n}^{i}_{N})\\
    \end{array}
    \right]\;.
$$ 


where $\sum_k p(m_k)=1$ and we have $12$ legal moves for Rukik. Such a function <span style="color:#f88146">**associates** encoded states with a discrete probability distribution of legal moves for **undoing** the scramble</span>. 

Again, training is performed offline (before the search for solving the Rubik Cube starts). 

**Inference**. During <span style="color:#f88146">the **search (test)**</span> we start at a vectorized space $\mathbf{n_0}$, corresponding to a scrabled cube very unlikely to be seen during training, and we query $f_{\theta}(\mathbf{n}_0)$. The result is $12$ probabilities, one per legal move. Then, from $\mathbf{n}_0$ <span style="color:#f88146">we **expand** $12$ candidates to un-scramble the Rubik Cube from it</span>:  


$$
\mathbf{n}'\in {\cal N}_{\mathbf{n}_0}=\left\{
    \begin{array}{cc}
    \mathbf{n}_0\circ m_1&\text{with prob.}\;\; p(m_1|\mathbf{n}_0)\\
    \mathbf{n}_0\circ m_2&\text{with prob.}\;\; p(m_2|\mathbf{n}_0)\\
    \ldots\\
    \mathbf{n}_0\circ m_{12}&\text{with prob.}\;\; p(m_{12}|\mathbf{n}_0)\\
    \end{array}
    \right\}\;
$$ 

where $\mathbf{n}'=\mathbf{n}_0\circ m_k$ is the state of $\Omega$ obtained after applying the move $m_k$ on $\mathbf{n}_0$. 

However, when expanding say $\mathbf{n}_1=\mathbf{n}_0\circ m_1$ we have that its probabilities are

$$
\mathbf{n}'\in {\cal N}_{\mathbf{n}_1}=\left\{
    \begin{array}{cc}
    \mathbf{n}_1\circ m_1&\text{with prob.}\;\; p(m_1|\mathbf{n}_1)p(m_1|\mathbf{n}_0)\\
    \mathbf{n}_1\circ m_2&\text{with prob.}\;\; p(m_2|\mathbf{n}_1)p(m_1|\mathbf{n}_0)\\
    \ldots\\
    \mathbf{n}_1\circ m_{12}&\text{with prob.}\;\; p(m_{12}|\mathbf{n}_1)p(m_1|\mathbf{n}_0)\\
    \end{array}
    \right\}\;.
$$

In other words, <span style="color:#f88146">the probability of each new state is the **product of past probabilities**</span>. This product becomes the $g(\mathbf{n})$ of a BFS strategy and there is no $h(\mathbf{n})$. 

### Cross Entropy 
Before analyzing the search in more detail, it is key to describe sucintly the learning of $f_{\theta}$. For the sake of simplicity, we consider that:

- We have only two classes (moves): $0$ and $1$. 
- The states of $\Omega$ are denoted by $\mathbf{x}_1,\mathbf{x}_2,\ldots$. 
- We do now know their distribution $p(\mathbf{x})$, because, in practice, the size of $\Omega$ is virtually infinite.
- However, *for the training set*, we know: 

$$
\begin{aligned}
\text{True Labels}:\;\; & p(m=0|\mathbf{x})\;\;\text{as well as}\;\; p(m=1|\mathbf{x}) = 1 - p(m=0|\mathbf{x})\\
\text{Predicted Labels}:\;\; & q_{\theta}(m=0|\mathbf{x})\;\;\text{as well as}\;\; q_{\theta}(m=1|\mathbf{x})= 1 - q_{\theta}(m=0|\mathbf{x})\\
\end{aligned}
$$

Where $q_{\theta}(m|\mathbf{x})$ is the distribution of the predictor (oracle) for a given configuration of the parameters $\theta$.  



The "cost" of the configuration $\theta$ is quantified by a **loss function**. The most used loss is the **cross-entropy** loss $CE$. For a given $\mathbf{x}$ we have: 

$$
\begin{aligned}
CE(\mathbf{x})&=-\sum_{c\in\{0,1\}}p(m=c|\mathbf{x})\log q_{\theta}(m=c|\mathbf{x})\\
              &=\sum_{c\in\{0,1\}}p(m=c|\mathbf{x})\log\frac{1}{q_{\theta}(m=c|\mathbf{x})}\\
              &=E\left(\log\frac{1}{q_{\theta}(m=c|\mathbf{x})}\right)\\
              
\end{aligned}
$$

or more understandable...

$$
CE(\mathbf{x})=-p(m=0|\mathbf{x})\log q_{\theta}(m=0|\mathbf{x}) - \underbrace{p(m=1|\mathbf{x})}_{1-p(m=0|\mathbf{x})}\log \underbrace{q_{\theta}(m=1|\mathbf{x})}_{1-q_{\theta}(m=0|\mathbf{x})}\;.\\
$$

<br></br>
<span style="color:#d94f0b"> 
**Example**. Given four examples $\mathbf{x}_1$, $\mathbf{x}_2$, $\mathbf{x}_3$ and $\mathbf{x}_4$, we have
</span>
<br></br>
<span style="color:#d94f0b">
$
\begin{aligned}
p(m=0|\mathbf{x}_1) = 1\;&\; p(m=1|\mathbf{x}_1)=0 \\
p(m=0|\mathbf{x}_2) = 0\;&\; p(m=1|\mathbf{x}_2)=1 \\
p(m=0|\mathbf{x}_3) = 1\;&\; p(m=1|\mathbf{x}_3)=0 \\
p(m=0|\mathbf{x}_4) = 0\;&\; p(m=1|\mathbf{x}_4)=1 \\
\end{aligned}
$
</span>
<br></br>
<span style="color:#d94f0b"> 
and for the configuration $\theta$ we have: 
</span>
<br></br>
<span style="color:#d94f0b">
$
\begin{aligned}
q_{\theta}(m=0|\mathbf{x}_1) = 0.9\;&\; q_{\theta}(m=1|\mathbf{x}_1)=0.1 \\
q_{\theta}(m=0|\mathbf{x}_2) = 0.2\;&\; q_{\theta}(m=1|\mathbf{x}_2)=0.8 \\
q_{\theta}(m=0|\mathbf{x}_3) = 0.7\;&\; q_{\theta}(m=1|\mathbf{x}_3)=0.3 \\
q_{\theta}(m=0|\mathbf{x}_4) = 0.3\;&\; q_{\theta}(m=1|\mathbf{x}_4)=0.7 \\
\end{aligned}
$
</span>
<br></br>
<span style="color:#d94f0b"> 
Then, their respective $CE$s are
</span>
<br></br>
<span style="color:#d94f0b">
$
\begin{aligned}
CE(\mathbf{x}_1) &= \mathbf{-1\cdot\log 0.9} - 0\cdot\log 0.1 &= 0.10 \\
CE(\mathbf{x}_2) &= -0\cdot\log 0.2 \mathbf{- 1\cdot\log 0.8} &= 0.22 \\
CE(\mathbf{x}_3) &= \mathbf{-1\cdot\log 0.7} - 0\cdot\log 0.3 &= 0.35 \\
CE(\mathbf{x}_4) &= -0\cdot\log 0.3 \mathbf{- 1\cdot\log 0.7} &= 0.35 \\
\end{aligned}
$
</span>
<br></br>
<span style="color:#d94f0b"> 
where the only significant distribution (in bold) is when we match the true distrution of each data. 
</span>
<br></br>
<span style="color:#d94f0b"> 
Following the above results, the best "fitted" points are $\mathbf{x}_3$ and $\mathbf{x}_4$ (see {numref}`CE-toy`).
</span>
<br></br>

```{figure} ./images/Topic3/CE-toy-removebg-preview.png
---
name: CE-toy
width: 500px
align: center
height: 400px
---
CEs of four examples (blue color is class $0$ and orange is class $1$). 
```

For a larger example, we have that when the configuration $\theta$ of the predictor is not good, then $q_{\theta}(m=1|\mathbf{x})$ and $q_{\theta}(m=0|\mathbf{x})$ **overlap** significantly (see {numref}`Overlap`). 

```{figure} ./images/Topic3/Overlap-removebg-preview.png
---
name: Overlap
width: 500px
align: center
height: 400px
---
Overlap of the predicted distributions. 
```

As a result, CEs are not so good but at some examples (see {numref}`Overlap-CEs`): 

```{figure} ./images/Topic3/Overlap-CEs-removebg-preview.png
---
name: Overlap-CEs
width: 500px
align: center
height: 400px
---
CEs for overlaped predicted distributions. 
```

Note that $CE$ has the form of a KL divergence between $p$ (only known for the training set) and $q$ (the distribution learnt by the predictor). Actually, look at the ratios: 

$$
\log\frac{1}{q_{\theta}(m=c|\mathbf{x})}\;.
$$

These ratios mean the log-likelihood of  the truth $1$ wrt of the prediction $q$. 

Actually, we have

$$
\begin{align}
E\left(\log\frac{1}{q}\right)&=\sum_{c\in\{0,1\}}p(c)\log\frac{1}{q(c)}\\
&= \sum_{c\in\{0,1\}}p(c)\log\frac{1}{q(c)}\cdot\frac{p(c)}{p(c)}\\
&=\sum_{c\in\{0,1\}}p(c)\log\frac{1}{p(c)} + \sum_{c\in\{0,1\}}p(c)\log\frac{p(c)}{q(c)}\\
&= H(p) + D(p||q)\;.
\end{align}
$$

Therefore, if we change $\theta$ to minimize $\frac{1}{N}\sum_{\mathbf{x}}CE(\mathbf{x})$, where $N$ is the number of training examples, we are implicitly minimizing the KL divergence between the predictor and the true distribution! Please, see a detailed discussion in the [Towards Datascience Article](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a/).

### Rubik State Space
The minimization of the average CE, now for $12$ classes (moves in Rubik), is as in {numref}`CE-Rubik`: 

```{figure} ./images/Topic3/CE-Rubik-removebg-preview.png
---
name: CE-Rubik
width: 500px
align: center
height: 500px
---
CEs for overlaped predicted distributions. 
```

Note, that as learning progresses, the CE curve flattens. Then, close to the optimum, $\theta$ is quasi stable and we can look at the structure of the state space $\Omega$. We attend to the last $100$ iterations and we proceed as follows: 

1) During these last iterations, we have explored $|\Omega'| =23,644$ states $\mathbf{n}_i$ as a "surrogate" of $\Omega$. 

2) The states in $\Omega'$ almost distributed uniformly. Actually, most of them are **visited once**: $23,201$ of $23,644$ ($98\%$). However, the remaining $2\%$ show that $\Omega'$ follows a **power law** (see {numref}`Power-Rubik`).

3) Move probabilities are almost uniform $p(m_k)=1/12=0.83$, as well. 

4) What about conditional probabilities $p_{\theta}(m_k|\mathbf{n}_i)$? Well, we observe that the predictor is **absolutely certain** about the move to recommend: $p_{\theta}(m_k|\mathbf{n}_i)=1$ for a given $m_k$ and $0$ for the remaiming moves. This behavior is consistent with learning the true distribution of data in $\Omega$ (see {numref}`Cond-Rubik`).

5) Finally, as it is expected, each move is equally reachable: $\sum_i p(m_k|\mathbf{n}_i)=1/12$ for all $m_k$ (see {numref}`Cum-Rubik`)

```{figure} ./images/Topic3/Power-Rubik-removebg-preview.png
---
name: Power-Rubik
width: 500px
align: center
height: 400px
---
Power-law of visited space for Rubik.
```

```{figure} ./images/Topic3/Rubik-Cond-removebg-preview.png
---
name: Cond-Rubik
width: 500px
align: center
height: 400px
---
Conditional probabilities in Rubik (for a sample of states).
```

```{figure} ./images/Topic3/Pred-Cumulative-removebg-preview.png
---
name: Cum-Rubik
width: 500px
align: center
height: 400px
---
Cummulative Conditional probabilities in Rubik per move.
```

### Beam Search and Rubik
**Beam Search**. Beam search is a particular case of BFS where the size of $\text{OPEN}$ is bounded, for instance to $2^k$ states. If at some point of the search we reach $2^k+1$ or more stats, <span style="color:#f88146">the $\text{OPEN}$ list is **purged** to retain only the best $2^k$ states</span>. In other words, at any moment, we keep up to the best $2^k$ states in $\text{OPEN}$.

**Self-supervised Rubik** is basically a beam search where $\text{OPEN}$ where we retain the best $2^k$ non-expanded nodes ordered in descending order wrt

$$
g(\mathbf{n})=p(\mathbf{n})\cdot p(\text{parent}(\mathbf{n}))
$$ 

and recursively, we define $p(\text{parent}(\mathbf{n}))$. In other words, the most promising sequence of moves is the one maximizing a product of probabilities $\prod_i p(i)$ starting and $p(\mathbf{n})$ and ending $p(\mathbf{n}_0)$. 

Actually, the deeper a path the less probable and more informative is. The above experiments, showing that $p(\mathbf{n})=1$ for only one of the $12$ moves, leads to a very focused search where few paths have $\prod_i p(i)\approx 1$ and the remaining ones have $\prod_i p(i)\approx 0$. 

**Effect of the Deep Oracle**. If the DNN is not good enouth, the Beam-Search algorithm is reduced to a bounded BFS. 

**Entropy Analysis**. Since $g(\mathbf{n})$ is defined in probabilistic terms, we can envision $\text{OPEN}$ as a probability distribution. In this regard, we *interpret beam search* as follows: 

1) During the first iterations $\text{OPEN}$ increments its entropy, i.e. the partial solutions become maximally diverse. 
2) As the search progresses, some partial paths (but not too much) are more likely than others. 
3) By the end of the search (close to the max-allowed-depth), the <span style="color:#f88146">entropy decreases only if the seach succeeds</span>. 

In {numref}`Entropy-Rubik`, we represent the solution length vs the average entropy of many executions (all of them successful). Note that:

1) Looking at the average solution length (between $24-26$), the vertical distribution of entropies is quite uniform (although medium-large entropies are more frequent than small ones). This is consistent with the fact that **the DNN $f_{\theta}$ becomes a nearly uniform sampler**. 

2) There is a **slight positive correlation** (0.28) between solution length and average entropy. The largest the required length the largest (and less diverse) the entropy. 

```{figure} ./images/Topic3/Entropy-Beam-removebg-preview.png
---
name: Entropy-Rubik
width: 500px
align: center
height: 400px
---
Entropy analysis for many executions of Rubik Beam Search.
```

**Power Law**. Obviously, a small value of $k$ in $2^k$ (max. size of $\text{OPEN}$) usually leads to poor solutions (we are sacrificing optimality to contain the combinatorial explosion). However, the experiments in [Self Supervised Rubik](https://openreview.net/pdf?id=bnBeNFB27b) show that as we move from $2^7$ to $2^{18}$, the Rubik solver improves significanlty. This is consistent with the **scaling law** used for Transformers.

```{figure} ./images/Topic3/Performance-DeepCube-removebg-preview.png
---
name: Performance-Rubik
width: 800px
align: center
height: 600px
---
Performance analysis of DeepCube with Beam Search (image from [Self Supervised Rubik](https://openreview.net/pdf?id=bnBeNFB27b)).
```

See also in {numref}`Performance-Rubik`, that the Rubik solver takes $N$ moves on average (where $N$ is God's number). However, there is a significant devation both up and below God's number!





## Appendix 
### Kullback-Leibler Divergence 
**Distances between distributions**. Consider two discrete random variables $X$ and $Y$ defined on the same domain ${\cal D}=\{z_1,z_2,\ldots,z_n\}$. Then we have 

$$
p_X(i) = p(X=z_i)\;\;\text{as well as}\;\; p_Y(i) = p(Y=z_i)\;\;\text{for}\; i=1,2,\ldots,n\;.
$$

Obviously, $\sum_i p_X(i) = \sum_i p_Y(i) = 1$. 

However, <span style="color:#f88146">how do we **measure a sort of distance** between $p_X$ and $p_Y$?</span> 

- First of all, consider $n$ as the **dimensionality** of the domain. 
- It [well known](https://homes.cs.washington.edu/%7Epedrod/papers/cacm12.pdf) that as $n$ increases and we generate data, the probabilistic mass is not uniform at all. For the multivariate Gaussian distribution, for instance, such a mass is in a shell around the mean. 

- In other words, high-dimensional data such as texts and images **do not live in a uniform (maximal entropy) space where everything is equally probable**. 

- Therefore, if $X$ is taken from "grey images of dogs" and $Y$ is taken from "grey images of cats" and $z_i\in [0,255]$ where $n=N\times N$ is the mumber of pixels, it is quite clear that $X$ and $Y$ **cannot be compared by means of an Euclidean norm**. 

The <span style="color:#f88146">**Kullback-Leibler Divergence** compares $p_X$ and $p_Y$ instead</span>. Again, the **Euclidean distance is not suitable** for comparing $p_X$ and $p_Y$ because it does not account for their intrinsic frequency and variability. 

<span style="color:#f88146">**Log-likelihood Ratio**</span>. The core of the Euclidean or Manhattan distance is $p_X(i) - p_y(i)$. However, given $i$ we have to dilucidate whether it comes from $p_X$ (dogs) or from $p_Y$ (cats). This leads to the following [Log-likelihood Statistical Test](https://en.wikipedia.org/wiki/Likelihood-ratio_test) 

$$
\begin{aligned}
H_0 &:\textbf{(null hypothesis):}\;i\;\text{is generated by}\; p_X\\ 
H_1 &:\textbf{(alternative hypothesis):}\;i\;\text{is generated by}\; p_Y\\ 
\end{aligned}
$$

where 

$$
\begin{aligned}
\text{If}\; \Lambda(i) \ge c,\;& \textbf{do not reject}\; H_0\;.\\
\text{If}\; \Lambda(i) < c,\;& \textbf{reject}\; H_0\;.\\
\end{aligned}
$$

and we have: 

$$
\Lambda(i) = \log\frac{p_X(i)}{p_Y(i)} = \log p_X(i) - \log p_Y(i):\;\;\textbf{log-likelihood}\;.
$$

Herein, the $\log$ is used in order to <span style="color:#f88146">maximize the likelihood</span>: the closer is $p$ to $1$ the smallest (less negative) is the $\log$. 

<span style="color:#f88146">**KL-divergence**</span>. Given $p_X$ and $p_Y$, we have the following *divergences*: 

$
\begin{aligned}
D(p_X||p_Y) &= \sum_i p_X(i)\log \frac{p_X(i)}{p_Y(i)}\ge 0\\
D(p_Y||p_X) &= \sum_i p_Y(i)\log \frac{p_Y(i)}{p_X(i)}\ge 0\;.\\
\end{aligned}
$

which can be seen as **expectations** of the corresponding log-ratios: respectively 

$
\begin{aligned}
D(p_X||p_Y) &= E\left(\log \frac{p_X(i)}{p_Y(i)}\right)\ge 0\\
D(p_Y||p_X) &= E\left(\log \frac{p_Y(i)}{p_X(i)}\right)\ge 0\;,\\
\end{aligned}
$

i.e. the <span style="color:#f88146">**KL divergence means** how good or bad goes the corresponing test on average</span>.

In general, $D(p_X||p_Y)\neq D(p_Y||p_X)$, since the triangular inequality 

$
D(p_X||p_Y) + D(p_Y||p_Z)\le D(p_X||p_Z)
$ 

is not verified. Then *we do not have a distance but a divergence*. Actually, both the Euclidean distance and the KL divergence belong to a wider family known as **Bregman Divergences** [Escolano et al, book. Chapter 7](https://link.springer.com/book/10.1007/978-1-84882-297-9). 


**KL-divergence for Bernouilli**. If $X\sim \text{Bernouilli}(p_X)$ and $Y\sim \text{Bernouilli}(p_Y)$, what is the form of $D(p_X||p_Y)$? 

Well, look that the **histogram** of a $\text{Bernouilli}(p)$ *does only have two bars*: $p$ and $1-p$, since $p + (1-p)=1$. Then, **we have two run two tests** when computing the KL-divergence:

$$
\begin{aligned}
D(p_X||p_Y)=p_X\log \frac{p_X}{p_Y} + (1-p_X)\log \frac{1-p_X}{1-p_Y}\;.
\end{aligned}
$$

<br></br>
<span style="color:#d94f0b"> 
**Example**. Compute the KL-divergence for Bernouilli distributions: $p_X=0.5$ and $p_Y=0.75$.
</span>
<span style="color:#d94f0b"> 
$
\begin{aligned}
D(p_X||p_Y)&=0.5\log \frac{0.5}{0.75} + 0.5\log \frac{0.5}{0.25}\\
           &=0.5\left(\log 0.5 - \log 0.75\right) +0.5\left(\log 0.5 - \log 0.25\right)\\
           &=0.5\cdot (-0,4) +0.5\cdot (+0.69)\\
           &=-0.2 + 0.34\\
           &= 0.14\;.
\end{aligned}
$
</span>
<span style="color:#d94f0b"> 
and 
</span>
<span style="color:#d94f0b"> 
$
\begin{aligned}
D(p_Y||p_X)&=0.75\log \frac{0.75}{0.5} + 0.25\log \frac{0.25}{0.5}\\
           &=0.75\left(\log 0.75 - \log 0.5\right) +0.25\left(\log 0.25 - \log 0.5\right)\\
           &=0.5\cdot (+0,4) +0.25\cdot (-0.69)\\
           &=0.2 - 0.1725\\
           &= 0.0275\;.
\end{aligned}
$
</span>
<br></br>
<span style="color:#d94f0b"> 
Therefore, $p_Y$ is closer to $p_X$ than $p_X$ is to $p_Y$!
</span>

**KL-divergence for Binomial and Normal**. Extending the above definition for comparing $X\sim \text{Binomial}(n,p_X)$ and $Y\sim \text{Binomial}(n,p_Y)$ we obtain: 

$$
\begin{aligned}
D(p_X||p_Y)=n\cdot p_X\log \frac{p_X}{p_Y} + n\cdot (1-p_X)\log \frac{1-p_X}{1-p_Y}\;.
\end{aligned}
$$

You can find the proof in [The Book of Statistical Proofs](https://statproofbook.github.io/P/bin-kl.html) but the **interpretation** is straightforward: since a Binomial variable is a sum of $n$ independent Bernouillis with the same probability of success, all we have to do is introduce $n$ in each summand of the KL-divergence. 

Obviuosly, the larger $n$ the larger the KL divergence! What about the KL for the Normal Distribution?. Well, the Normal/Gaussian distribution is continuous and the sum in the divergence must be replaced by an integral. From the [same book](https://statproofbook.github.io/P/norm-kl) we have, for $p_X={\cal N}(\mu_X,\sigma_X^2)$ and $p_Y={\cal N}(\mu_Y,\sigma_Y^2)$: 

$$
\begin{aligned}
D(p_X||p_Y)=\frac{1}{2}\left[\frac{(\mu_Y-\mu_X)^2}{\sigma_Y^2} + \frac{\sigma_X^2}{\sigma_Y^2}-\log\frac{\sigma_X^2}{\sigma_Y^2}-1\right]\;.
\end{aligned}
$$

Notably, if $\mu_X = \mu_Y$, the KL divergence relies only on the variances' ratio. This basically shows that statistical dispersion (aka of entropy) dominates how KL divergences are expressed. This explains why the <span style="color:#f88146">KL divergence is usually called the **relative entropy**</span>.

In {numref}`KL-Normal`, we explore the two cases (similar vs different mean). Note that co-centering the distributions while preserving the variances reduces dramatically the KL divergence. 

```{figure} ./images/Topic3/KL-normal-removebg-preview.png
---
name: KL-Normal
width: 800px
align: center
height: 400px
---
KL divergences between Normals with different and same mean. 
```
-->