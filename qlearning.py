import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import pickle
import os
import random
from collections import defaultdict

# Import your existing Playground class - make sure this file is in the same directory
# as your playground implementation
from playground import Playground, Car, Point2D

class QLearningAgent:
    def __init__(self, learning_rate=0.5, discount_factor=0.95, exploration_rate=0.5, 
                 exploration_decay=0.995, min_exploration_rate=0.0):
        """
        Initialize Q-Learning agent with parameters
        
        Args:
            learning_rate: Alpha - learning rate
            discount_factor: Gamma - discount factor for future rewards
            exploration_rate: Epsilon - initial exploration rate
            exploration_decay: Rate at which exploration decreases
            min_exploration_rate: Minimum exploration rate
        """
        # Define wheel angle range and increment
        self.wheel_min = -40
        self.wheel_max = 40
        self.wheel_increment = 40
        
        # Calculate number of actions based on 10-degree increments
        self.actions = list(range(self.wheel_min, self.wheel_max + 1, self.wheel_increment))
        self.n_actions = len(self.actions)
        
        # Agent learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table as a defaultdict for sparse representation
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        
        # State discretization parameters
        self.distance_buckets = [5, 10, 20, 40, float('inf')]  # Distance buckets
        self.angle_increment = 10  # Degrees per angle bucket
        
    def discretize_state(self, state, angle):
        """
        Convert continuous state values to discrete buckets for Q-table lookup
        
        Args:
            state: List of [front_dist, right_dist, left_dist]
            angle: Current facing angle of the car
            
        Returns:
            Tuple representing the discretized state
        """
        # Discretize distances
        front_dist, right_dist, left_dist = state
        
        # Handle -1 (no obstacle) case
        if front_dist == -1:
            front_bucket = len(self.distance_buckets) - 1
        else:
            front_bucket = next((i for i, threshold in enumerate(self.distance_buckets) 
                               if front_dist < threshold), len(self.distance_buckets) - 1)
            
        if right_dist == -1:
            right_bucket = len(self.distance_buckets) - 1
        else:
            right_bucket = next((i for i, threshold in enumerate(self.distance_buckets) 
                               if right_dist < threshold), len(self.distance_buckets) - 1)
            
        if left_dist == -1:
            left_bucket = len(self.distance_buckets) - 1
        else:
            left_bucket = next((i for i, threshold in enumerate(self.distance_buckets) 
                              if left_dist < threshold), len(self.distance_buckets) - 1)
        
        # Discretize angle (-90 to 270)
        angle_bucket = int((angle + 90) // self.angle_increment)
        
        # Return discretized state as a tuple (hashable for dict lookup)
        return (front_bucket, right_bucket, left_bucket, angle_bucket)
    
    def choose_action(self, state, angle):
        """
        Choose an action using epsilon-greedy policy
        
        Returns:
            int: Index of the selected action
        """
        # Discretize the state
        discrete_state = self.discretize_state(state, angle)
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Explore: choose a random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: choose the best action from Q-table
            return np.argmax(self.q_table[discrete_state])
    
    def get_wheel_angle(self, action_idx):
        """
        Convert action index to wheel angle value
        """
        return self.actions[action_idx]
    
    def get_action_index(self, wheel_angle):
        """
        Convert wheel angle to action index
        """
        # Find the closest valid action
        return self.actions.index(min(self.actions, key=lambda x: abs(x - wheel_angle)))
    
    def learn(self, state, angle, action, reward, next_state, next_angle, done):
        """
        Update Q-table using the Q-learning update rule
        """
        # Discretize current and next states
        current_discrete = self.discretize_state(state, angle)
        next_discrete = self.discretize_state(next_state, next_angle)
        
        # Current Q-value
        current_q = self.q_table[current_discrete][action]
        
        # Calculate target Q-value
        if done:
            # If terminal state, only consider immediate reward
            target_q = reward
        else:
            # Otherwise, include discounted future rewards
            max_next_q = np.max(self.q_table[next_discrete])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value using Q-learning rule
        self.q_table[current_discrete][action] += self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay
    
    def save_q_table(self, filename="q_table.pkl"):
        """
        Save the Q-table to a file
        """
        # Convert defaultdict to regular dict for serialization
        q_dict = dict(self.q_table)
        with open(filename, 'wb') as f:
            pickle.dump(q_dict, f)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename="q_table.pkl"):
        """
        Load Q-table from a file
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                q_dict = pickle.load(f)
                # Convert back to defaultdict
                self.q_table = defaultdict(lambda: np.zeros(self.n_actions), q_dict)
            print(f"Q-table loaded from {filename}")
            return True
        return False


def calculate_reward(env, done, prev_position=None):
    """
    Calculate reward based on the current state of the environment
    
    Args:
        env: The Playground environment
        done: Whether the episode is done
        prev_position: Previous car position for distance calculation
        
    Returns:
        float: The calculated reward
    """
    car_pos = env.car.getPosition()
    
    # Check if car reached destination (success)
    if done and env.reached_goal:
        return 100.0  # Large positive reward for reaching goal
    
    # Check if car crashed (failure)
    if done and not env.reached_goal:
        return -100.0  # Large negative reward for crashing
    
    # Calculate distance to goal (center of destination area)
    dest_center_x = (env.destination_topleft.x + env.destination_bottomright.x) / 2
    dest_center_y = (env.destination_topleft.y + env.destination_bottomright.y) / 2
    dest_center = Point2D(dest_center_x, dest_center_y)
    
    current_distance = car_pos.distToPoint2D(dest_center)
    
    # If we have a previous position, reward for getting closer to destination
    if prev_position:
        prev_distance = prev_position.distToPoint2D(dest_center)
        distance_reward = (prev_distance - current_distance) * 2.0
    else:
        distance_reward = 0.0
    
    # Small penalty for each step to encourage efficiency
    step_penalty = -0.1
    
    # Return combined reward
    return distance_reward + step_penalty


def train(episodes=1000, max_steps=1000, render_interval=100, save_interval=100):
    """
    Train the Q-learning agent
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        render_interval: How often to render the environment
        save_interval: How often to save the Q-table
    """
    # Initialize environment and agent
    env = Playground()
    agent = QLearningAgent()
    
    # Try to load existing Q-table if available
    agent.load_q_table()
    
    # Lists to track performance
    rewards_history = []
    steps_history = []  
    success_history = []
    
    # Setup visualization
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Track best performance
    best_reward = float('-inf')
    
    for episode in range(episodes):
        # Reset environment and get initial state
        state = env.reset()
        angle = env.car.angle
        total_reward = 0
        prev_position = None
        
        # Determine if we should render this episode
        should_render = (episode % render_interval == 0)
        
        for step in range(max_steps):
            # Get car position for reward calculation
            prev_position = env.car.getPosition()
            
            # Choose action and convert to wheel angle
            action_idx = agent.choose_action(state, angle)
            wheel_angle = agent.get_wheel_angle(action_idx)
            
            # Convert wheel angle to environment action
            # We need to map our wheel angle to the environment's action space
            env.car.setWheelAngle(wheel_angle)
            #action = env.calWheelAngleFromAction(action_idx)
            
            # Take action in environment
            next_state = env.step()
            next_angle = env.car.angle
            done = env.done
            
            # Calculate reward
            reward = calculate_reward(env, done, prev_position)
            total_reward += reward
            
            # Agent learns from the transition
            agent.learn(state, angle, action_idx, reward, next_state, next_angle, done)
            
            # Render if needed
            if should_render:
                env.render(ax1)
                # Add action info to plot
                action_text = f"Action: {wheel_angle}° wheel angle"
                ax1.text(0.05, 0.85, action_text, transform=ax1.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                plt.draw()
                plt.pause(0.01)
            
            # Update state and angle
            state = next_state
            angle = next_angle
            
            # Break if episode is done
            if done:
                break
        
        # Record episode statistics
        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(1 if env.reached_goal else 0)
        
        # Print episode info
        print(f"Episode {episode+1}/{episodes}, Steps: {step+1}, "
              f"Reward: {total_reward:.2f}, Success: {env.reached_goal}, "
              f"Exploration: {agent.exploration_rate:.4f}")
        
        # Plot performance metrics
        if episode > 0 and episode % 10 == 0:
            ax2.clear()
            # Plot rolling average reward
            window = 10
            if len(rewards_history) >= window:
                rolling_avg = [np.mean(rewards_history[max(0, i-window):i+1]) 
                              for i in range(len(rewards_history))]
                ax2.plot(rolling_avg, 'b-', label='Reward (10-ep rolling avg)')
            else:
                ax2.plot(rewards_history, 'b-', label='Reward')
            
            # Plot success rate (rolling average)
            if len(success_history) >= window:
                rolling_success = [np.mean(success_history[max(0, i-window):i+1]) * 100 
                                 for i in range(len(success_history))]
                ax2.plot(rolling_success, 'g-', label='Success % (10-ep rolling avg)')
            
            ax2.set_title('Training Performance')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Metrics')
            ax2.legend()
            plt.draw()
            plt.pause(0.01)
        
        # Save model at intervals
        if episode % save_interval == 0 and episode > 0:
            agent.save_q_table(f"q_table_ep{episode}.pkl")
        
        # Track and save best model
        avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else total_reward
        if avg_reward > best_reward and episode >= 10:
            best_reward = avg_reward
            agent.save_q_table("q_table_best.pkl")
    
    # Final save
    agent.save_q_table()
    plt.ioff()
    
    # Plot final performance graphs
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(steps_history)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 3)
    # Calculate success rate over time (rolling window)
    window = 50
    if len(success_history) >= window:
        rolling_success = [np.mean(success_history[max(0, i-window):i+1]) * 100 
                         for i in range(len(success_history))]
        plt.plot(rolling_success)
        plt.title(f'Success Rate (%) - {window}-episode Rolling Average')
    else:
        plt.plot(np.cumsum(success_history) / (np.arange(len(success_history)) + 1) * 100)
        plt.title('Success Rate (%) - Cumulative Average')
    plt.xlabel('Episode')
    plt.ylabel('Success %')
    
    plt.subplot(2, 2, 4)
    # Calculate rolling average reward
    if len(rewards_history) >= window:
        rolling_avg = [np.mean(rewards_history[max(0, i-window):i+1]) 
                      for i in range(len(rewards_history))]
        plt.plot(rolling_avg)
        plt.title(f'Reward - {window}-episode Rolling Average')
    else:
        plt.plot(rewards_history)
        plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.show()


def evaluate(episodes=10, max_steps=1000, q_table_file="q_table_best.pkl"):
    """
    Evaluate a trained agent
    """
    # Initialize environment and agent
    env = Playground()
    agent = QLearningAgent()
    
    # Load Q-table
    if not agent.load_q_table(q_table_file):
        print(f"Could not load Q-table from {q_table_file}. Please train first.")
        return
    
    # Set exploration rate to 0 for pure exploitation
    agent.exploration_rate = 0.0
    
    # Setup visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Track statistics
    success_count = 0
    total_steps = 0
    total_rewards = 0
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        angle = env.car.angle
        episode_reward = 0
        prev_position = None
        
        for step in range(max_steps):
            # Render the environment
            env.render(ax)
            plt.draw()
            plt.pause(0.05)  # Slower for better visualization
            
            # Store position for reward calculation
            prev_position = env.car.getPosition()
            
            # Choose best action based on learned Q-values
            action_idx = agent.choose_action(state, angle)
            wheel_angle = agent.get_wheel_angle(action_idx)
            
            
            # Convert wheel angle to environment action
            env.car.setWheelAngle(wheel_angle)
            #action = env.calWheelAngleFromAction(action_idx)
            
            # Add action info to plot
            action_text = f"Action: {wheel_angle}° wheel angle"
            ax.text(0.05, 0.85, action_text, transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Take action
            next_state = env.step()
            next_angle = env.car.angle
            done = env.done
            
            # Calculate reward (for tracking only)
            reward = calculate_reward(env, done, prev_position)
            episode_reward += reward
            
            # Update state and angle
            state = next_state
            angle = next_angle
            
            # Break if episode is done
            if done:
                break
        
        # Update statistics
        total_steps += step + 1
        total_rewards += episode_reward
        if env.reached_goal:
            success_count += 1
        
        # Print episode results
        print(f"Evaluation Episode {episode+1}/{episodes}, Steps: {step+1}, "
              f"Reward: {episode_reward:.2f}, Success: {env.reached_goal}")
        
        # Final render of the episode
        env.render(ax)
        plt.draw()
        plt.pause(1.0)  # Pause to see final state
    
    # Print overall evaluation results
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_count/episodes*100:.2f}%")
    print(f"Average Steps: {total_steps/episodes:.2f}")
    print(f"Average Reward: {total_rewards/episodes:.2f}")
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Q-Learning for Self-Driving Car")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--episodes', type=int, default=1000, 
                        help='Number of episodes')
    parser.add_argument('--render_interval', type=int, default=100,
                        help='Render every N episodes during training')
    parser.add_argument('--q_table', type=str, default='q_table_best.pkl',
                        help='Q-table file for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(episodes=args.episodes, render_interval=args.render_interval)
    else:
        evaluate(episodes=args.episodes, q_table_file=args.q_table)