import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import sys
import numpy as np
import time

# Import the custom modules (assuming they're in the same directory)
from qlearning import train, evaluate, QLearningAgent, Playground
from playground import Point2D

class QlearningUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning Self-Driving Car Simulator")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.training_thread = None
        self.evaluation_thread = None
        self.stop_threads = False
        
        # For tracking best performance
        self.best_avg_reward = float('-inf')
        
        # For tracking car path
        self.car_path_x = []
        self.car_path_y = []
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top control panel
        self.create_control_panel()
        
        # Create main display area with two subplots
        self.create_plot_area()
        
        # Create bottom status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Training/Evaluation params and results
        self.training_progress = 0
        self.train_episode_count = 0
        self.train_success_count = 0
        self.train_rewards = []
        self.train_steps = []
        self.train_success_rate = []
        
        # Setup matplotlib event handler for closing
        plt.rcParams['figure.max_open_warning'] = 0

    def create_control_panel(self):
        # Create control panel frame
        control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Training parameters section
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # First row: Training parameters
        ttk.Label(params_frame, text="Episodes:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.episodes_var = tk.IntVar(value=200)
        ttk.Spinbox(params_frame, from_=1, to=10000, textvariable=self.episodes_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Max Steps:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.max_steps_var = tk.IntVar(value=1000)
        ttk.Spinbox(params_frame, from_=1, to=10000, textvariable=self.max_steps_var, width=8).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Render Interval:").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.render_interval_var = tk.IntVar(value=20)
        ttk.Spinbox(params_frame, from_=1, to=100, textvariable=self.render_interval_var, width=8).grid(row=0, column=5, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Save Interval:").grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        self.save_interval_var = tk.IntVar(value=100)
        ttk.Spinbox(params_frame, from_=1, to=500, textvariable=self.save_interval_var, width=8).grid(row=0, column=7, padx=5, pady=5)
        
        # Second row: Agent parameters
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.learning_rate_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(params_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.learning_rate_var, width=8).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Discount Factor:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.discount_factor_var = tk.DoubleVar(value=0.95)
        ttk.Spinbox(params_frame, from_=0.1, to=1.0, increment=0.01, textvariable=self.discount_factor_var, width=8).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Exploration Rate:").grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)
        self.exploration_rate_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.exploration_rate_var, width=8).grid(row=1, column=5, padx=5, pady=5)
        
        # Third row: Q-table file selection
        ttk.Label(params_frame, text="Q-Table File:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.qtable_file_var = tk.StringVar(value="q_table_best.pkl")
        ttk.Entry(params_frame, textvariable=self.qtable_file_var, width=30).grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(params_frame, text="Browse", command=self.browse_qtable).grid(row=2, column=4, padx=5, pady=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Control buttons
        self.train_btn = ttk.Button(buttons_frame, text="Start Training", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.evaluate_btn = ttk.Button(buttons_frame, text="Start Evaluation", command=self.start_evaluation)
        self.evaluate_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(buttons_frame, text="Stop", command=self.stop_running, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.exit_btn = ttk.Button(buttons_frame, text="Exit", command=self.on_closing)
        self.exit_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

    def create_plot_area(self):
        # Create a frame for plots
        plot_frame = ttk.Frame(self.main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a figure for matplotlib plots
        self.fig = plt.figure(figsize=(12, 8))
        
        # Create two subplots
        self.simulation_ax = self.fig.add_subplot(1, 2, 1)
        self.simulation_ax.set_title("Simulation Environment")
        self.simulation_ax.set_xlabel("X Position")
        self.simulation_ax.set_ylabel("Y Position")
        
        self.stats_ax = self.fig.add_subplot(1, 2, 2)
        self.stats_ax.set_title("Training Statistics")
        self.stats_ax.set_xlabel("Episode")
        self.stats_ax.set_ylabel("Value")
        
        # Embed the matplotlib figure in the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for training/evaluation results
        results_frame = ttk.LabelFrame(self.main_frame, text="Results")
        results_frame.pack(fill=tk.X, pady=5)
        
        # Results grid
        results_grid = ttk.Frame(results_frame)
        results_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(results_grid, text="Episodes:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.result_episodes_var = tk.StringVar(value="0/0")
        ttk.Label(results_grid, textvariable=self.result_episodes_var).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(results_grid, text="Success Rate:").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.result_success_var = tk.StringVar(value="0.0%")
        ttk.Label(results_grid, textvariable=self.result_success_var).grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(results_grid, text="Avg. Reward:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.result_reward_var = tk.StringVar(value="0.0")
        ttk.Label(results_grid, textvariable=self.result_reward_var).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(results_grid, text="Avg. Steps:").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        self.result_steps_var = tk.StringVar(value="0.0")
        ttk.Label(results_grid, textvariable=self.result_steps_var).grid(row=1, column=3, padx=5, pady=2, sticky=tk.W)

    def browse_qtable(self):
        filename = filedialog.askopenfilename(
            title="Select Q-Table File",
            filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*"))
        )
        if filename:
            self.qtable_file_var.set(filename)

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Already Running", "Training is already running!")
            return
        
        # Disable buttons while training
        self.train_btn.config(state=tk.DISABLED)
        self.evaluate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Reset progress and counters
        self.progress_var.set(0)
        self.train_episode_count = 0
        self.train_success_count = 0
        self.train_rewards = []
        self.train_steps = []
        self.train_success_rate = []
        self.stop_threads = False
        
        # Reset car path tracking
        self.car_path_x = []
        self.car_path_y = []
        
        # Get parameters
        episodes = self.episodes_var.get()
        max_steps = self.max_steps_var.get()
        render_interval = self.render_interval_var.get()
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self.training_process,
            args=(episodes, max_steps, render_interval)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
        # Update status
        self.status_var.set("Training started...")

    def training_process(self, episodes, max_steps, render_interval):
        """Run the training process in a separate thread"""
        try:
            # Initialize environment and agent
            env = Playground()
            agent = QLearningAgent(
                learning_rate=self.learning_rate_var.get(),
                discount_factor=self.discount_factor_var.get(),
                exploration_rate=self.exploration_rate_var.get()
            )
            
            # Try to load existing Q-table if available
            agent.load_q_table()
            
            # Lists to track performance
            rewards_history = []
            steps_history = []
            success_history = []
            
            for episode in range(episodes):
                if self.stop_threads:
                    break
                
                # Update episode counter on UI thread
                self.train_episode_count = episode + 1
                self.root.after(0, self.update_progress, episode + 1, episodes)
                
                # Reset environment and get initial state
                state = env.reset()
                angle = env.car.angle
                total_reward = 0
                prev_position = None
                
                # Reset car path for this episode
                self.car_path_x = []
                self.car_path_y = []
                
                # Determine if we should render this episode
                should_render = (episode % render_interval == 0)
                
                for step in range(max_steps):
                    if self.stop_threads:
                        break
                    
                    # Track car position for the path
                    car_pos = env.car.getPosition()
                    self.car_path_x.append(car_pos.x)
                    self.car_path_y.append(car_pos.y)
                    
                    # Get car position for reward calculation
                    prev_position = env.car.getPosition()
                    
                    # Choose action and convert to wheel angle
                    action_idx = agent.choose_action(state, angle)
                    wheel_angle = agent.get_wheel_angle(action_idx)
                    
                    # Set wheel angle
                    env.car.setWheelAngle(wheel_angle)
                    
                    # Take action in environment
                    next_state = env.step()
                    next_angle = env.car.angle
                    done = env.done
                    
                    # Calculate reward
                    reward = self.calculate_reward(env, done, prev_position)
                    total_reward += reward
                    
                    # Agent learns from the transition
                    agent.learn(state, angle, action_idx, reward, next_state, next_angle, done)
                    
                    # Render if needed
                    if should_render:
                        # We need to update the plot on the main thread
                        self.root.after(0, self.update_simulation_plot, env)
                    
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
                
                if env.reached_goal:
                    self.train_success_count += 1
                
                # Calculate success rate
                success_rate = self.train_success_count / self.train_episode_count * 100
                
                # Store data for plotting
                self.train_rewards = rewards_history
                self.train_steps = steps_history
                self.train_success_rate.append(success_rate)
                
                # Update stats every 5 episodes
                if episode % 5 == 0 or episode == episodes - 1:
                    self.root.after(0, self.update_stats_plot, rewards_history, success_history)
                
                # Save Q-table at regular intervals
                save_interval = self.save_interval_var.get()  # Use the user-defined save interval
                if (episode + 1) % save_interval == 0:
                    save_filename = f"q_table_ep{episode+1}.pkl"
                    agent.save_q_table(save_filename)
                    self.root.after(0, self.update_status, f"Saved Q-table to {save_filename}")
                    time.sleep(0.5)  # Brief pause to show the save message
                
                # Also save the best model if this is the best performance so far
                if episode >= 10:  # Wait for at least 10 episodes
                    window_size = min(10, len(rewards_history))
                    current_avg_reward = np.mean(rewards_history[-window_size:])
                    
                    # Keep track of best average reward
                    if not hasattr(self, 'best_avg_reward') or current_avg_reward > self.best_avg_reward:
                        self.best_avg_reward = current_avg_reward
                        agent.save_q_table("q_table_best.pkl")
                        self.root.after(0, self.update_status, f"New best performance! Saved to q_table_best.pkl")
                        time.sleep(0.5)  # Brief pause to show the save message
                
                # Update results display every episode
                avg_reward = np.mean(rewards_history[-min(10, len(rewards_history)):])
                avg_steps = np.mean(steps_history[-min(10, len(steps_history)):])
                self.root.after(0, self.update_results, 
                               f"{episode+1}/{episodes}",
                               f"{success_rate:.2f}%",
                               f"{avg_reward:.2f}",
                               f"{avg_steps:.2f}")
                
                # Status update
                status_msg = f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}, Success: {env.reached_goal}"
                self.root.after(0, self.update_status, status_msg)
                
                # Slight pause to prevent UI freezing
                time.sleep(0.01)
            
            # Save final Q-table
            agent.save_q_table()
            
            # Final status update
            self.root.after(0, self.update_status, "Training complete!")
            
            # Re-enable buttons
            self.root.after(0, self.enable_buttons)
            
        except Exception as e:
            error_msg = f"Error in training: {str(e)}"
            self.root.after(0, self.show_error, error_msg)
            self.root.after(0, self.enable_buttons)

    def start_evaluation(self):
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            messagebox.showinfo("Already Running", "Evaluation is already running!")
            return
        
        # Disable buttons while evaluating
        self.train_btn.config(state=tk.DISABLED)
        self.evaluate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Reset progress
        self.progress_var.set(0)
        self.stop_threads = False
        
        # Reset car path tracking
        self.car_path_x = []
        self.car_path_y = []
        
        # Get parameters
        episodes = min(10, self.episodes_var.get())  # Use fewer episodes for evaluation
        max_steps = self.max_steps_var.get()
        q_table_file = self.qtable_file_var.get()
        
        # Start evaluation in a separate thread
        self.evaluation_thread = threading.Thread(
            target=self.evaluation_process,
            args=(episodes, max_steps, q_table_file)
        )
        self.evaluation_thread.daemon = True
        self.evaluation_thread.start()
        
        # Update status
        self.status_var.set("Evaluation started...")

    def evaluation_process(self, episodes, max_steps, q_table_file):
        """Run the evaluation process in a separate thread"""
        try:
            # Initialize environment and agent
            env = Playground()
            agent = QLearningAgent()
            
            # Set exploration rate to 0 for pure exploitation
            agent.exploration_rate = 0.0
            
            # Load Q-table
            if not agent.load_q_table(q_table_file):
                self.root.after(0, self.show_error, f"Could not load Q-table from {q_table_file}. Please train first.")
                self.root.after(0, self.enable_buttons)
                return
            
            # Track statistics
            success_count = 0
            total_steps = 0
            total_rewards = 0
            
            for episode in range(episodes):
                if self.stop_threads:
                    break
                
                # Update progress on UI thread
                self.root.after(0, self.update_progress, episode + 1, episodes)
                
                # Reset environment
                state = env.reset()
                angle = env.car.angle
                episode_reward = 0
                prev_position = None
                
                # Reset car path for this episode
                self.car_path_x = []
                self.car_path_y = []
                
                for step in range(max_steps):
                    if self.stop_threads:
                        break
                    
                    # Track car position for the path
                    car_pos = env.car.getPosition()
                    self.car_path_x.append(car_pos.x)
                    self.car_path_y.append(car_pos.y)
                    
                    # Update plot on UI thread
                    self.root.after(0, self.update_simulation_plot, env)
                    
                    # Store position for reward calculation
                    prev_position = env.car.getPosition()
                    
                    # Choose best action based on learned Q-values
                    action_idx = agent.choose_action(state, angle)
                    wheel_angle = agent.get_wheel_angle(action_idx)
                    
                    # Set wheel angle
                    env.car.setWheelAngle(wheel_angle)
                    
                    # Take action
                    next_state = env.step()
                    next_angle = env.car.angle
                    done = env.done
                    
                    # Calculate reward (for tracking only)
                    reward = self.calculate_reward(env, done, prev_position)
                    episode_reward += reward
                    
                    # Update state and angle
                    state = next_state
                    angle = next_angle
                    
                    # Brief pause for visualization
                    time.sleep(0.05)
                    
                    # Break if episode is done
                    if done:
                        break
                
                # Update statistics
                total_steps += step + 1
                total_rewards += episode_reward
                if env.reached_goal:
                    success_count += 1
                
                # Update results display
                success_rate = success_count / (episode + 1) * 100
                avg_reward = total_rewards / (episode + 1)
                avg_steps = total_steps / (episode + 1)
                
                self.root.after(0, self.update_results, 
                               f"{episode+1}/{episodes}",
                               f"{success_rate:.2f}%",
                               f"{avg_reward:.2f}",
                               f"{avg_steps:.2f}")
                
                # Status update
                status_msg = f"Eval Episode {episode+1}/{episodes}, Steps: {step+1}, Reward: {episode_reward:.2f}, Success: {env.reached_goal}"
                self.root.after(0, self.update_status, status_msg)
                
                # Final render of the episode
                self.root.after(0, self.update_simulation_plot, env)
                time.sleep(1.0)  # Pause to see final state
            
            # Final status update
            final_msg = f"Evaluation complete! Success Rate: {success_count/episodes*100:.2f}%"
            self.root.after(0, self.update_status, final_msg)
            
            # Re-enable buttons
            self.root.after(0, self.enable_buttons)
            
        except Exception as e:
            error_msg = f"Error in evaluation: {str(e)}"
            self.root.after(0, self.show_error, error_msg)
            self.root.after(0, self.enable_buttons)

    def calculate_reward(self, env, done, prev_position=None):
        """
        Calculate reward based on the current state of the environment
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

    def stop_running(self):
        """Stop any running training or evaluation"""
        self.stop_threads = True
        self.status_var.set("Stopping... Please wait.")

    def update_progress(self, current, total):
        """Update the progress bar"""
        progress = int(current / total * 100)
        self.progress_var.set(progress)

    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)

    def update_results(self, episodes, success_rate, avg_reward, avg_steps):
        """Update the results display"""
        self.result_episodes_var.set(episodes)
        self.result_success_var.set(success_rate)
        self.result_reward_var.set(avg_reward)
        self.result_steps_var.set(avg_steps)

    def update_simulation_plot(self, env):
        """Update the simulation plot with the current environment state and car path"""
        self.simulation_ax.clear()
        
        # First, render the environment
        env.render(self.simulation_ax)
        
        # Then, draw the car's path if we have points
        if len(self.car_path_x) > 1:
            self.simulation_ax.plot(self.car_path_x, self.car_path_y, 'm-', alpha=0.7, linewidth=2, 
                                   label='Car Path')
        
        # Add a legend if path is being displayed
        if len(self.car_path_x) > 0:
            self.simulation_ax.legend(loc='lower right')
            
        self.canvas.draw()

    def update_stats_plot(self, rewards, successes):
        """Update the statistics plot"""
        self.stats_ax.clear()
        
        # Plot training metrics
        episode_numbers = list(range(1, len(rewards) + 1))
        
        # Calculate rolling averages
        window = min(10, len(rewards))
        if len(rewards) >= window:
            rolling_reward = [np.mean(rewards[max(0, i-window):i+1]) 
                             for i in range(len(rewards))]
            rolling_success = [np.mean(successes[max(0, i-window):i+1]) * 100 
                              for i in range(len(successes))]
            
            # Plot the metrics
            self.stats_ax.plot(episode_numbers, rolling_reward, 'b-', label='Reward (10-ep rolling)')
            self.stats_ax.plot(episode_numbers, rolling_success, 'g-', label='Success % (10-ep rolling)')
        else:
            self.stats_ax.plot(episode_numbers, rewards, 'b-', label='Reward')
            success_rate = [np.mean(successes[:i+1]) * 100 for i in range(len(successes))]
            self.stats_ax.plot(episode_numbers, success_rate, 'g-', label='Success %')
        
        self.stats_ax.set_title('Training Statistics')
        self.stats_ax.set_xlabel('Episode')
        self.stats_ax.set_ylabel('Value')
        self.stats_ax.legend()
        
        self.canvas.draw()

    def enable_buttons(self):
        """Re-enable UI buttons after completion"""
        self.train_btn.config(state=tk.NORMAL)
        self.evaluate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def show_error(self, message):
        """Display an error message"""
        messagebox.showerror("Error", message)

    def on_closing(self):
        """Handle window closing"""
        # Stop any running processes
        self.stop_threads = True
        
        # Wait a moment for threads to stop
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=1.0)
        
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=1.0)
        
        # Close plt figures
        plt.close('all')
        
        # Close the window
        self.root.destroy()
        sys.exit(0)