#!/usr/bin/env python3
"""
Main entry point for the autonomous delivery agent system.
Provides a CLI interface for running different pathfinding algorithms.
"""

import argparse
import sys
import os
import time
import json
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment import GridEnvironment, Position
from algorithms.bfs import BFSPlanner, BFSWithCosts
from algorithms.ucs import UCSPlanner, UCSPlannerOptimized
from algorithms.astar import AStarPlanner, AStarWeighted
from algorithms.local_search import LocalSearchPlanner, SimulatedAnnealingPlanner


class AutonomousDeliveryAgent:
    """Main agent class that coordinates pathfinding algorithms."""
    
    def __init__(self):
        self.environment = None
        self.algorithms = {
            'bfs': BFSPlanner,
            'bfs_cost': BFSWithCosts,
            'ucs': UCSPlanner,
            'ucs_opt': UCSPlannerOptimized,
            'astar': AStarPlanner,
            'astar_weighted': AStarWeighted,
            'local': LocalSearchPlanner,
            'sa': SimulatedAnnealingPlanner
        }
    
    def load_environment(self, map_file: str):
        """Load environment from map file."""
        self.environment = GridEnvironment(0, 0)  # Will be resized by load_from_file
        self.environment.load_from_file(map_file)
        
        # Add dynamic obstacle schedules for dynamic.txt
        if 'dynamic' in map_file:
            self._setup_dynamic_obstacles()
    
    def _setup_dynamic_obstacles(self):
        """Set up dynamic obstacles for the dynamic map."""
        # Create moving vehicles with deterministic schedules
        schedules = {
            'A': [(i, Position(5 + i % 10, 10)) for i in range(20)],
            'B': [(i, Position(15, 5 + i % 10)) for i in range(20)],
            'C': [(i, Position(10 - i % 10, 15)) for i in range(20)],
        }
        
        for vehicle_id, schedule in schedules.items():
            self.environment.add_dynamic_obstacle_schedule(vehicle_id, schedule)
    
    def run_algorithm(self, algorithm_name: str, start: Position, goal: Position, 
                     time_step: int = 0, **kwargs) -> Dict[str, Any]:
        """Run a specific algorithm and return results."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Create planner instance
        if algorithm_name == 'astar_weighted':
            planner = self.algorithms[algorithm_name](self.environment, weight=kwargs.get('weight', 1.5))
        elif algorithm_name in ['local', 'sa']:
            planner = self.algorithms[algorithm_name](self.environment, **kwargs)
        else:
            planner = self.algorithms[algorithm_name](self.environment)
        
        # Run algorithm
        path = planner.find_path(start, goal, time_step)
        
        # Get statistics
        stats = planner.get_statistics()
        stats['algorithm'] = algorithm_name
        stats['path_found'] = path is not None
        stats['path'] = [str(pos) for pos in path] if path else None
        
        return stats
    
    def compare_algorithms(self, start: Position, goal: Position, time_step: int = 0) -> Dict[str, Dict[str, Any]]:
        """Compare all algorithms on the same problem."""
        results = {}
        
        print(f"Comparing algorithms from {start} to {goal} at time step {time_step}")
        print("=" * 60)
        
        for alg_name in ['bfs', 'ucs', 'astar', 'local']:
            try:
                print(f"Running {alg_name}...")
                result = self.run_algorithm(alg_name, start, goal, time_step)
                results[alg_name] = result
                
                print(f"  Path found: {result['path_found']}")
                if result['path_found']:
                    print(f"  Cost: {result['path_cost']:.2f}")
                    print(f"  Nodes expanded: {result['nodes_expanded']}")
                    print(f"  Time: {result['execution_time']:.4f}s")
                print()
                
            except Exception as e:
                print(f"  Error: {e}")
                results[alg_name] = {'error': str(e)}
                print()
        
        return results
    
    def benchmark(self, output_file: str = None):
        """Run comprehensive benchmark tests."""
        print("Running benchmark tests...")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            {'map': 'maps/small.txt', 'start': (0, 0), 'goal': (9, 9), 'name': 'small_map'},
            {'map': 'maps/medium.txt', 'start': (0, 0), 'goal': (19, 19), 'name': 'medium_map'},
            {'map': 'maps/large.txt', 'start': (0, 0), 'goal': (49, 49), 'name': 'large_map'},
            {'map': 'maps/dynamic.txt', 'start': (0, 0), 'goal': (19, 19), 'name': 'dynamic_map'},
        ]
        
        all_results = {}
        
        for config in test_configs:
            print(f"\nTesting {config['name']}...")
            
            # Load environment
            self.load_environment(config['map'])
            start = Position(config['start'][0], config['start'][1])
            goal = Position(config['goal'][0], config['goal'][1])
            
            # Run comparison
            results = self.compare_algorithms(start, goal)
            all_results[config['name']] = {
                'map_file': config['map'],
                'start': config['start'],
                'goal': config['goal'],
                'results': results
            }
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        
        return all_results
    
    def visualize_path(self, algorithm_name: str, start: Position, goal: Position, 
                      time_step: int = 0, **kwargs):
        """Visualize the path found by an algorithm."""
        result = self.run_algorithm(algorithm_name, start, goal, time_step, **kwargs)
        
        if result['path_found']:
            print(f"Path found by {algorithm_name}:")
            print(f"Cost: {result['path_cost']:.2f}")
            print(f"Nodes expanded: {result['nodes_expanded']}")
            print(f"Execution time: {result['execution_time']:.4f}s")
            print()
            
            # Convert string positions back to Position objects for visualization
            path_positions = []
            for pos_str in result['path']:
                x, y = map(int, pos_str.strip('()').split(', '))
                path_positions.append(Position(x, y))
            
            self.environment.visualize(agent_pos=start, path=path_positions)
        else:
            print(f"No path found by {algorithm_name}")
    
    def dynamic_replanning_demo(self, algorithm_name: str = 'local'):
        """Demonstrate dynamic replanning with moving obstacles."""
        print(f"Dynamic replanning demo with {algorithm_name}")
        print("=" * 60)
        
        # Load dynamic environment
        self.load_environment('maps/dynamic.txt')
        start = Position(0, 0)
        goal = Position(19, 19)
        
        print("Initial state:")
        self.environment.visualize(agent_pos=start)
        
        # Simulate agent movement with replanning
        current_pos = start
        total_cost = 0
        time_step = 0
        
        while current_pos != goal and time_step < 20:
            print(f"\nTime step {time_step}: Agent at {current_pos}")
            
            # Find path from current position
            result = self.run_algorithm(algorithm_name, current_pos, goal, time_step)
            
            if not result['path_found'] or len(result['path']) < 2:
                print("No path found or reached goal!")
                break
            
            # Move one step along the path
            next_pos_str = result['path'][1]  # Skip current position
            x, y = map(int, next_pos_str.strip('()').split(', '))
            next_pos = Position(x, y)
            
            # Check if next position is still valid
            if self.environment.is_obstacle(next_pos, time_step + 1):
                print(f"Path blocked! Replanning...")
                # Advance time and try again
                time_step += 1
                self.environment.advance_time()
                continue
            
            # Move to next position
            current_pos = next_pos
            total_cost += result.get('path_cost', 1)
            time_step += 1
            self.environment.advance_time()
            
            print(f"Moved to {current_pos}, total cost: {total_cost}")
            
            # Show environment state
            self.environment.visualize(agent_pos=current_pos)
        
        if current_pos == goal:
            print(f"\nGoal reached! Total cost: {total_cost}, Time steps: {time_step}")
        else:
            print(f"\nFailed to reach goal. Final position: {current_pos}")


def parse_position(pos_str: str) -> Position:
    """Parse position string in format 'x,y'."""
    try:
        x, y = map(int, pos_str.split(','))
        return Position(x, y)
    except ValueError:
        raise ValueError(f"Invalid position format: {pos_str}. Use 'x,y' format.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Autonomous Delivery Agent Pathfinding System')
    
    # Required arguments
    parser.add_argument('--algorithm', '-a', choices=['bfs', 'bfs_cost', 'ucs', 'ucs_opt', 'astar', 'astar_weighted', 'local', 'sa'],
                       help='Algorithm to use')
    parser.add_argument('--map', '-m', help='Map file path')
    parser.add_argument('--start', '-s', help='Start position (format: x,y)')
    parser.add_argument('--goal', '-g', help='Goal position (format: x,y)')
    
    # Optional arguments
    parser.add_argument('--time-step', '-t', type=int, default=0, help='Time step for dynamic obstacles')
    parser.add_argument('--visualize', '-v', action='store_true', help='Enable visualization')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare all algorithms')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Run benchmark tests')
    parser.add_argument('--dynamic-demo', '-d', action='store_true', help='Run dynamic replanning demo')
    parser.add_argument('--output', '-o', help='Output file for results')
    parser.add_argument('--weight', type=float, default=1.5, help='Weight for weighted A*')
    
    args = parser.parse_args()
    
    # Create agent
    agent = AutonomousDeliveryAgent()
    
    try:
        if args.benchmark:
            # Run benchmark
            results = agent.benchmark(args.output or 'benchmark_results.json')
            
        elif args.dynamic_demo:
            # Run dynamic demo
            algorithm = args.algorithm or 'local'
            agent.dynamic_replanning_demo(algorithm)
            
        elif args.compare:
            # Compare algorithms
            if not all([args.map, args.start, args.goal]):
                print("Error: --map, --start, and --goal are required for comparison")
                return 1
            
            agent.load_environment(args.map)
            start = parse_position(args.start)
            goal = parse_position(args.goal)
            
            results = agent.compare_algorithms(start, goal, args.time_step)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
                
        else:
            # Run single algorithm
            if not all([args.algorithm, args.map, args.start, args.goal]):
                print("Error: --algorithm, --map, --start, and --goal are required")
                return 1
            
            agent.load_environment(args.map)
            start = parse_position(args.start)
            goal = parse_position(args.goal)
            
            if args.visualize:
                agent.visualize_path(args.algorithm, start, goal, args.time_step, weight=args.weight)
            else:
                result = agent.run_algorithm(args.algorithm, start, goal, args.time_step, weight=args.weight)
                print(f"Algorithm: {result['algorithm']}")
                print(f"Path found: {result['path_found']}")
                if result['path_found']:
                    print(f"Cost: {result['path_cost']:.2f}")
                    print(f"Nodes expanded: {result['nodes_expanded']}")
                    print(f"Execution time: {result['execution_time']:.4f}s")
                    print(f"Path: {' -> '.join(result['path'])}")
                else:
                    print("No path found!")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
