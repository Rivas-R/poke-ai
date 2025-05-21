"""
Pokemon Pinball - Training module (Optimized)
"""
from pyboy import PyBoy
import torch
import time
import os
import numpy as np

# Import from optimized modules
from pinball_common import ROM_PATH, MODELS_DIR, STATE_SIZE, ACTION_SIZE, ADDR_BALL_X, ADDR_BALL_Y, preprocess_state, apply_action, init_game
from pinball_agent import PinballAgent

def train_pinball_agent(num_episodes=500, batch_size=64, max_frames_per_episode=10000, skip_frames=6):
    """
    Train the pinball agent using reinforcement learning (optimized version)
    """
    # Initialize PyBoy
    pyboy = PyBoy(ROM_PATH, sound_volume=0)
    pyboy.set_emulation_speed(0)  # Maximum speed
    
    # Initialize agent
    agent = PinballAgent(STATE_SIZE, ACTION_SIZE)
    
    print(f"Initialized agent with state size: {STATE_SIZE} and action size: {ACTION_SIZE}")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Variables for tracking
    scores = []
    frame_counts = []
    
    print(f"Starting training for {num_episodes} episodes...")
    total_start_time = time.time()
    
    # Main training loop
    for episode in range(num_episodes):
        # Reset game
        init_game(pyboy)
        
        # Get initial state
        last_pos_x = current_pos_x = pyboy.memory[ADDR_BALL_X]
        last_pos_y = current_pos_y = pyboy.memory[ADDR_BALL_Y]

        state = preprocess_state(pyboy, current_pos_x, current_pos_y, last_pos_x, last_pos_y)
        
        total_reward = 0
        last_score = pyboy.game_wrapper.score
        last_balls = pyboy.game_wrapper.balls_left
        last_s_balls = pyboy.game_wrapper.lost_ball_during_saver
        last_pokemon_caught = pyboy.game_wrapper.pokemon_caught_in_session
        
        # Log this episode
        print(f"Episode {episode+1}/{num_episodes}, Epsilon: {agent.epsilon:.4f}")
        episode_start_time = time.time()
        
        # Episode frame loop
        for frame in range(max_frames_per_episode):
            # Select and execute action
            action = agent.act(state)
            
            # Advance game multiple frames - increased skip frames for faster training
            for _ in range(skip_frames):
                apply_action(pyboy, action)
                pyboy.tick()
                if pyboy.game_wrapper.game_over:
                    break
            
            # Get new state
            current_pos_x = pyboy.memory[ADDR_BALL_X]
            current_pos_y = pyboy.memory[ADDR_BALL_Y]
            next_state = preprocess_state(pyboy, current_pos_x, current_pos_y, last_pos_x, last_pos_y)
            
            # Calculate reward
            current_score = pyboy.game_wrapper.score
            reward = current_score - last_score  # Reward based on score 
            
            if current_pos_x == last_pos_x and current_pos_y == last_pos_y:
                reward -= 50 # Small penalty for doing nothing

            current_balls = pyboy.game_wrapper.balls_left
            if current_balls < last_balls:
                reward -= 1000  # Penalty for losing a ball
            elif current_balls > last_balls:
                reward += 3000  # Reward for getting a ball
            
            current_s_balls = pyboy.game_wrapper.lost_ball_during_saver
            if current_s_balls > last_s_balls:
                reward -= 5000   # Penalty for ball lost during saver

            current_pokemon_caught = pyboy.game_wrapper.pokemon_caught_in_session
            if current_pokemon_caught > last_pokemon_caught:
                reward += 1000000 # reward for catching a pokemon
            
            # Check if game is over
            done = pyboy.game_wrapper.game_over
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train the network with batches of experience
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                
            # Update for next iteration
            last_pos_x = current_pos_x
            last_pos_y = current_pos_y
            state = next_state
            total_reward += reward
            last_score = current_score
            last_balls = current_balls
            last_s_balls = current_s_balls
            last_pokemon_caught = current_pokemon_caught
            
            # End episode if game is over
            if done:
                break
        
        # Episode statistics
        duration = time.time() - episode_start_time
        scores.append(total_reward)
        frame_counts.append(frame)
        
        print(f"  Score: {pyboy.game_wrapper.score}")
        print(f"  Duration: {duration:.2f} seconds, Frames: {frame}")
        print(f"  Total reward: {total_reward:.2f}")
        
        # Save model periodically - reduced frequency to save time
        if (episode + 1) % 100 == 0 or episode == num_episodes - 1:
            model_path = os.path.join(MODELS_DIR, f"pinball_model_ep{episode+1}.pth")
            torch.save(agent.model.state_dict(), model_path)
            print(f"  Model saved: {model_path}")
            
            # Report progress
            elapsed = time.time() - total_start_time
            estimated_total = elapsed / (episode + 1) * num_episodes
            remaining = estimated_total - elapsed
            print(f"  Progress: {episode+1}/{num_episodes} episodes, {elapsed:.2f}s elapsed, ~{remaining:.2f}s remaining")
    
    # Save final model
    final_model_path = os.path.join(MODELS_DIR, "pinball_model_final.pth")
    torch.save(agent.model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Close PyBoy
    pyboy.stop()
    
    total_duration = time.time() - total_start_time
    print("\nTraining completed!")
    print(f"Total training time: {total_duration:.2f} seconds")
    print(f"Final scores: {scores[-10:]}")
    print(f"Average of last 10 scores: {sum(scores[-10:]) / 10}")
    
    return scores, frame_counts