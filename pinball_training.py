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
        stuck_counter = 0
        
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
            
            reward = 0
    
            # 1. Score-based reward (normalized)
            current_score = pyboy.game_wrapper.score
            score_gain = current_score - last_score
            reward += score_gain * 0.01  # Scale down score rewards
            
            # 2. Ball management (moderate penalties/rewards)
            current_balls = pyboy.game_wrapper.balls_left
            if current_balls < last_balls:
                reward -= 500  # Significant but not overwhelming penalty
            elif current_balls > last_balls:
                reward += 1000  # Good reward for extra ball
            
            # 3. Saver ball management
            current_s_balls = pyboy.game_wrapper.lost_ball_during_saver
            if current_s_balls > last_s_balls:
                reward -= 300  # Moderate penalty
            
            # 4. Pokemon catching (scaled down but still significant)
            current_pokemon_caught = pyboy.game_wrapper.pokemon_caught_in_session
            if current_pokemon_caught > last_pokemon_caught:
                reward += 5000  # Large reward but not game-breaking
            
            # 5. Movement/activity reward (encourage active play)
            movement = abs(current_pos_x - last_pos_x) + abs(current_pos_y - last_pos_y)
            if movement > 10:  # Ball is moving significantly
                reward += 1  # Small positive reinforcement for active play
            
            # 6. Improved stuck penalty (more responsive)
            if abs(current_pos_x - last_pos_x) < 3 and abs(current_pos_y - last_pos_y) < 3:
                stuck_penalty = min(stuck_counter * 2, 50)  # Grows faster, caps at reasonable level
                reward -= stuck_penalty
                stuck_counter = min(stuck_counter + 1, 25)
            else:
                stuck_counter = max(stuck_counter - 1, 0)  # Decay stuck counter when moving
            
            # 7. Flipper timing reward (if you can detect good flipper hits)
            # This would require additional game state analysis
            # if good_flipper_timing:
            #     reward += 5
            
            # 8. Keep ball alive reward (small continuous reward)
            if not pyboy.game_wrapper.game_over:
                reward += 0.1  # Small reward for each frame ball stays alive
            
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
            

            #print(f'stuck: {int(stuck)}  {stuck}')
            #print(f"{current_pos_x}, {current_pos_y}")
            # End episode if game is over
            if done:
                break
        
        # Episode statistics
        duration = time.time() - episode_start_time
        scores.append(total_reward)
        frame_counts.append(frame)
        
        print(f"  Score: {pyboy.game_wrapper.score:,}")
        print(f"  Duration: {duration:.2f} seconds, Frames: {frame}")
        print(f"  Total reward: {total_reward:.2f}")
        
        # Save model periodically - reduced frequency to save time
        if (episode + 1) % 100 == 0 or episode == num_episodes - 1:
            model_path = os.path.join(MODELS_DIR, f"gen8_ep{episode+1}.pth")
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