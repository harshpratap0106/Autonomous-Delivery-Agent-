#!/usr/bin/env python3
"""
Demo script for the autonomous delivery agent system.
Demonstrates key features and capabilities.
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import AutonomousDeliveryAgent
from environment import Position


def run_demo():
    """Run a comprehensive demo of the system."""
    print("=" * 60)
    print("AUTONOMOUS DELIVERY AGENT - DEMONSTRATION")
    print("=" * 60)
    
    agent = AutonomousDeliveryAgent()
    
    # Demo 1: Basic pathfinding
    print("\n1. BASIC PATHFINDING DEMO")
    print("-" * 30)
    
    agent.load_environment('maps/small.txt')
    start = Position(0, 0)
    goal = Position(9, 9)
    
    print(f"Finding path from {start} to {goal} using A*...")
    result = agent.run_algorithm('astar', start, goal)
    
    if result['path_found']:
        print(f"✓ Path found! Cost: {result['path_cost']:.2f}")
        print(f"  Nodes expanded: {result['nodes_expanded']}")
        print(f"  Execution time: {result['execution_time']:.4f}s")
        print(f"  Path length: {len(result['path'])} steps")
    else:
        print("✗ No path found!")
    
    # Demo 2: Algorithm comparison
    print("\n2. ALGORITHM COMPARISON DEMO")
    print("-" * 30)
    
    print("Comparing BFS, UCS, A*, and Local Search...")
    results = agent.compare_algorithms(start, goal)
    
    print("\nResults Summary:")
    for alg_name, result in results.items():
        if result.get('path_found', False):
            print(f"  {alg_name.upper()}: Cost={result['path_cost']:.2f}, "
                  f"Nodes={result['nodes_expanded']}, Time={result['execution_time']:.4f}s")
        else:
            print(f"  {alg_name.upper()}: No path found")
    
    # Demo 3: Dynamic replanning
    print("\n3. DYNAMIC REPLANNING DEMO")
    print("-" * 30)
    
    print("Demonstrating dynamic replanning with moving obstacles...")
    print("(This will show how the agent adapts to changing environment)")
    
    # Run a shorter version of the dynamic demo
    agent.load_environment('maps/dynamic.txt')
    start = Position(0, 0)
    goal = Position(9, 9)  # Shorter path for demo
    
    print(f"Starting dynamic demo from {start} to {goal}...")
    
    current_pos = start
    total_cost = 0
    time_step = 0
    max_steps = 10  # Limit for demo
    
    while current_pos != goal and time_step < max_steps:
        print(f"\nTime step {time_step}: Agent at {current_pos}")
        
        # Find path from current position
        result = agent.run_algorithm('local', current_pos, goal, time_step)
        
        if not result['path_found'] or len(result['path']) < 2:
            print("No path found or reached goal!")
            break
        
        # Move one step along the path
        next_pos_str = result['path'][1]  # Skip current position
        x, y = map(int, next_pos_str.strip('()').split(', '))
        next_pos = Position(x, y)
        
        # Check if next position is still valid
        if agent.environment.is_obstacle(next_pos, time_step + 1):
            print(f"Path blocked! Replanning...")
            time_step += 1
            agent.environment.advance_time()
            continue
        
        # Move to next position
        current_pos = next_pos
        total_cost += result.get('path_cost', 1)
        time_step += 1
        agent.environment.advance_time()
        
        print(f"Moved to {current_pos}, total cost: {total_cost}")
    
    if current_pos == goal:
        print(f"\n✓ Goal reached! Total cost: {total_cost}, Time steps: {time_step}")
    else:
        print(f"\nDemo completed. Final position: {current_pos}")
    
    # Demo 4: Visualization
    print("\n4. VISUALIZATION DEMO")
    print("-" * 30)
    
    print("Showing environment visualization...")
    agent.load_environment('maps/small.txt')
    
    # Find a path to visualize
    result = agent.run_algorithm('astar', Position(0, 0), Position(9, 9))
    
    if result['path_found']:
        print("Environment with optimal path:")
        agent.environment.visualize(agent_pos=Position(0, 0))
        
        # Convert string positions back to Position objects for visualization
        path_positions = []
        for pos_str in result['path']:
            x, y = map(int, pos_str.strip('()').split(', '))
            path_positions.append(Position(x, y))
        
        print("Path found by A*:")
        for i, pos in enumerate(path_positions):
            print(f"  Step {i}: {pos}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo run more detailed experiments:")
    print("  python main.py --benchmark")
    print("  python main.py --compare --map maps/medium.txt --start 0,0 --goal 19,19")
    print("  python main.py --dynamic-demo")
    print("\nTo run individual algorithms:")
    print("  python main.py --algorithm astar --map maps/small.txt --start 0,0 --goal 9,9 --visualize")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Make sure all dependencies are installed and maps are available.")
        sys.exit(1)
