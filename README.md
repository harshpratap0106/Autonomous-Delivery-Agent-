# Autonomous-Delivery-Agent-
Project Overview:
This project, part of the CSA2001 Fundamentals of AI and ML course, involves designing and implementing an autonomous delivery agent. This agent navigates a 2D grid-based city to deliver packages efficiently. The environment includes static obstacles, varying terrain costs, and dynamic obstacles, challenging the agent to find optimal paths while adhering to constraints like time and fuel.

The core of the project is to implement and compare various search algorithms to solve this pathfinding problem.

Features of this agent:
Grid-Based Environment: The city is modeled as a 2D grid, where each cell represents different terrains, obstacles, or key locations.

Rational Agent: The agent is designed to be rational, selecting actions that maximize its performance measure (e.g., minimizing cost, time, or fuel).

Diverse Pathfinding Algorithms:

Uninformed Search: Breadth-First Search (BFS) and Uniform-Cost Search (UCS).

Informed Search: A* search with an admissible heuristic (Manhattan distance).

Local Search / Replanning: Hill-climbing with random restarts to adapt to dynamic obstacles.

Dynamic Obstacle Handling: The agent can replan its path when it encounters unexpected or moving obstacles.

Comprehensive Evaluation: The performance of each algorithm is compared based on path cost, nodes expanded, and execution time.

Environment Model:
Grid: A 2D array represents the city map.

Cells: Each cell has an integer movement cost ≥ 1.

1: Standard terrain (e.g., road)

>1: Difficult terrain (e.g., mud, hills)

0 or -1: Impassable static obstacle (e.g., building)

Agent: Occupies a single cell and can move to adjacent cells (up, down, left, right).

Dynamic Obstacles: These are moving obstacles that occupy a cell at a specific time. Their movements can be deterministic (predictable) or unpredictable.

Algorithms Implemented:
Breadth-First Search (BFS): An uninformed search algorithm that explores all neighbor nodes at the present depth before moving on to the nodes at the next depth level. It guarantees finding the shortest path in terms of the number of steps, but not necessarily the lowest cost.

Uniform-Cost Search (UCS): A variation of Dijkstra's algorithm. It explores the graph by expanding the node with the lowest path cost from the start. It guarantees finding the path with the minimum total cost.

A* Search: An informed search algorithm that uses a heuristic to guide its search towards the goal. It combines the cost to reach the node (g(n)) with an estimated cost to the goal (h(n)). We use the Manhattan distance as our admissible heuristic, as it never overestimates the actual cost on a grid where only 4-directional movement is allowed.

Hill-Climbing with Random Restarts: A local search algorithm used for replanning. If the agent's path is blocked by a new dynamic obstacle, it will attempt to find a new local path around it. If it gets stuck in a local minimum, it can restart the search from a different state.
