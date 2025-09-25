import argparse
import heapq
import time
import os

# --- Helper Classes ---

class Node:
    """A node class for the search algorithms."""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost from this node to goal
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

class GridCity:
    """Models the 2D grid environment."""
    def __init__(self, filename):
        self.grid = self.load_map(filename)
        self.width = len(self.grid[0])
        self.height = len(self.grid)
        self.start = None
        self.goal = None
        self.moving_obstacle_path = self.get_moving_obstacle_path()
        self.find_start_and_goal()

    def load_map(self, filename):
        """Loads a map from a text file."""
        grid = []
        with open(filename, 'r') as f:
            for line in f:
                grid.append(list(line.strip()))
        return grid

    def find_start_and_goal(self):
        """Locates the start and goal positions on the grid."""
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 'A':
                    self.start = (x, y)
                elif cell == 'D':
                    self.goal = (x, y)

    def get_cost(self, position):
        """Returns the movement cost for a given cell."""
        x, y = position
        cell = self.grid[y][x]
        if cell == '.':
            return 1
        elif cell == '~':
            return 5
        elif cell == '#':
            return float('inf')  # Obstacles are impassable
        return 1

    def is_valid(self, position):
        """Checks if a position is within grid bounds and not an obstacle."""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] != '#'
        return False
    
    def get_moving_obstacle_path(self):
        """
        Generates a deterministic path for a moving obstacle.
        A simple back-and-forth path is used for demonstration.
        """
        path = []
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 'M':
                    # Assuming only one moving obstacle for simplicity
                    # Generates a path moving right and then back left
                    path.append((x, y))
                    for i in range(1, 10):
                        path.append((x + i, y))
                    for i in range(9, 0, -1):
                        path.append((x + i, y))
                    return path
        return []

    def get_moving_obstacle_position(self, time_step):
        """Returns the moving obstacle's position at a given time step."""
        if not self.moving_obstacle_path:
            return None
        # Loop through the path
        return self.moving_obstacle_path[time_step % len(self.moving_obstacle_path)]


# --- Search Algorithms ---

def bfs_search(env):
    """Breadth-First Search for the shortest path in steps."""
    start_node = Node(None, env.start)
    end_node = Node(None, env.goal)

    frontier = [start_node]
    explored = set()

    start_time = time.time()
    nodes_expanded = 0

    while frontier:
        current_node = frontier.pop(0)
        nodes_expanded += 1
        explored.add(current_node.position)

        if current_node == end_node:
            end_time = time.time()
            return reconstruct_path(current_node), nodes_expanded, end_time - start_time

        neighbors = get_neighbors(env, current_node.position)
        for next_pos in neighbors:
            if next_pos not in explored:
                new_node = Node(current_node, next_pos)
                frontier.append(new_node)
                explored.add(next_pos)
    
    end_time = time.time()
    return None, nodes_expanded, end_time - start_time # Path not found

def uniform_cost_search(env):
    """Uniform-Cost Search for the least cost path."""
    start_node = Node(None, env.start)
    end_node = Node(None, env.goal)

    frontier = [(0, start_node)]  # (cost, node)
    explored = set()

    start_time = time.time()
    nodes_expanded = 0

    while frontier:
        cost, current_node = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_node.position in explored:
            continue
        explored.add(current_node.position)

        if current_node == end_node:
            end_time = time.time()
            return reconstruct_path(current_node), nodes_expanded, end_time - start_time

        neighbors = get_neighbors(env, current_node.position)
        for next_pos in neighbors:
            neighbor_cost = env.get_cost(next_pos)
            if neighbor_cost != float('inf'):
                new_node = Node(current_node, next_pos)
                new_node.g = cost + neighbor_cost
                heapq.heappush(frontier, (new_node.g, new_node))

    end_time = time.time()
    return None, nodes_expanded, end_time - start_time # Path not found

def a_star_search(env):
    """A* search for the optimal path using a heuristic."""
    start_node = Node(None, env.start)
    end_node = Node(None, env.goal)

    frontier = []
    start_node.h = manhattan_distance(start_node.position, end_node.position)
    start_node.f = start_node.g + start_node.h
    heapq.heappush(frontier, (start_node.f, start_node))

    explored = set()

    start_time = time.time()
    nodes_expanded = 0

    while frontier:
        f_cost, current_node = heapq.heappop(frontier)
        nodes_expanded += 1

        if current_node.position in explored:
            continue
        explored.add(current_node.position)

        if current_node == end_node:
            end_time = time.time()
            return reconstruct_path(current_node), nodes_expanded, end_time - start_time

        neighbors = get_neighbors(env, current_node.position)
        for next_pos in neighbors:
            neighbor_cost = env.get_cost(next_pos)
            if neighbor_cost != float('inf'):
                new_node = Node(current_node, next_pos)
                new_node.g = current_node.g + neighbor_cost
                new_node.h = manhattan_distance(new_node.position, end_node.position)
                new_node.f = new_node.g + new_node.h
                heapq.heappush(frontier, (new_node.f, new_node))

    end_time = time.time()
    return None, nodes_expanded, end_time - start_time # Path not found

def hill_climbing_replan(env):
    """
    A simplified hill-climbing replanning strategy.
    
    This simulation demonstrates dynamic replanning by having the agent
    move along a pre-calculated path. If a "new" obstacle appears, it
    will re-plan its path from its current position.
    
    Note: A true hill-climbing algorithm would explore a state space. Here,
    we're using a greedy approach where the "best" next step is chosen
    based on a simple heuristic, which is a common application of the
    hill-climbing concept in pathfinding.
    """
    start_time = time.time()
    nodes_expanded = 0
    
    print("Initial planning with A*...")
    path, _, _ = a_star_search(env)
    
    if not path:
        print("No initial path found.")
        return None, 0, 0
    
    current_pos = env.start
    path_cost = 0
    replan_count = 0
    
    for i in range(len(path) - 1):
        next_pos = path[i+1]
        
        # Simulate a moving obstacle appearing after a few steps
        if i == 5 and (env.grid[next_pos[1]][next_pos[0]] == 'M'):
            print(f"\nTime step {i}: Encountered dynamic obstacle at {next_pos}. Replanning...")
            replan_count += 1
            
            # Create a temporary environment with the new obstacle
            temp_env = GridCity(map_file)
            temp_env.start = current_pos
            
            # Re-plan from the current position
            new_path, expanded, _ = a_star_search(temp_env)
            nodes_expanded += expanded
            
            if new_path:
                print("New path found. Continuing...")
                path = path[:i+1] + new_path[1:]
                next_pos = new_path[1]
            else:
                print("Could not find a new path. Agent is stuck.")
                end_time = time.time()
                return None, nodes_expanded, end_time - start_time
        
        path_cost += env.get_cost(next_pos)
        current_pos = next_pos
        print(f"Time step {i+1}: Moved to {current_pos}. Path cost: {path_cost}")
        
    end_time = time.time()
    return path, nodes_expanded, end_time - start_time

# --- Helper Functions ---

def manhattan_distance(pos1, pos2):
    """Calculates the Manhattan distance heuristic."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_neighbors(env, position):
    """Returns a list of valid 4-connected neighbors for a given position."""
    x, y = position
    neighbors = [
        (x, y - 1),  # Up
        (x, y + 1),  # Down
        (x - 1, y),  # Left
        (x + 1, y)   # Right
    ]
    return [pos for pos in neighbors if env.is_valid(pos)]

def reconstruct_path(current_node):
    """Reconstructs the path from the end node back to the start."""
    path = []
    while current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
    return path[::-1] # Reverse the path

def print_results(path, path_cost, nodes_expanded, time_taken):
    """Prints the results of the search."""
    if path:
        print("\n--- Results ---")
        print(f"Path found: {path}")
        print(f"Path cost: {path_cost}")
        print(f"Nodes expanded: {nodes_expanded}")
        print(f"Time taken: {time_taken:.4f} seconds")
    else:
        print("\nNo path found.")

# --- Main Function ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Delivery Agent Planner")
    parser.add_argument("command", choices=["bfs", "ucs", "a_star", "replan"],
                        help="The search algorithm to use.")
    parser.add_argument("--map", dest="map_file", required=True,
                        help="Path to the map file.")
    
    args = parser.parse_args()
    map_file = args.map_file

    if not os.path.exists(map_file):
        print(f"Error: Map file '{map_file}' not found.")
    else:
        env = GridCity(map_file)
        path = None
        nodes_expanded = 0
        time_taken = 0.0

        if args.command == "bfs":
            print(f"Running Breadth-First Search on '{map_file}'...")
            path, nodes_expanded, time_taken = bfs_search(env)
            path_cost = sum(env.get_cost(pos) for pos in path) if path else 0
            print_results(path, path_cost, nodes_expanded, time_taken)
            
        elif args.command == "ucs":
            print(f"Running Uniform-Cost Search on '{map_file}'...")
            path, nodes_expanded, time_taken = uniform_cost_search(env)
            path_cost = sum(env.get_cost(pos) for pos in path) if path else 0
            print_results(path, path_cost, nodes_expanded, time_taken)

        elif args.command == "a_star":
            print(f"Running A* Search on '{map_file}'...")
            path, nodes_expanded, time_taken = a_star_search(env)
            path_cost = sum(env.get_cost(pos) for pos in path) if path else 0
            print_results(path, path_cost, nodes_expanded, time_taken)
            
        elif args.command == "replan":
            print(f"Running Dynamic Replanning on '{map_file}'...")
            path, nodes_expanded, time_taken = hill_climbing_replan(env)
            path_cost = sum(env.get_cost(pos) for pos in path) if path else 0
            print_results(path, path_cost, nodes_expanded, time_taken)
