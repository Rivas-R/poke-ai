"""
Pokemon Pinball - Play using trained model (Optimized)
"""
import random
from pyboy import PyBoy
import torch
import time
import os

# Import from optimized modules
from pinball_common import (
    ROM_PATH, STATE_SIZE, ACTION_SIZE, ADDR_BALL_X, ADDR_BALL_Y, SAVE_STATE_PATH,
    preprocess_state, apply_action, init_game
)
from pinball_agent import PinballDQN

def play_pinball_with_model(model_path, render=True, max_frames=20000, skip_frames=4):
    """
    Play Pokemon Pinball using a trained model
    
    Args:
        model_path: Path to the trained model file
        render: If True, set normal speed for watching, otherwise use max speed
        max_frames: Maximum number of frames to play
        skip_frames: Number of frames to skip between actions
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Initialize PyBoy
    pyboy = PyBoy(ROM_PATH)
    
    if render:
        pyboy.set_emulation_speed(1)  # Normal speed for watching
    else:
        pyboy.set_emulation_speed(0)  # Max speed if not watching
    
    # Initialize the game
    game_wrapper = init_game(pyboy)
    
    # Load the model
    model = PinballDQN(STATE_SIZE, ACTION_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded model with state size: {STATE_SIZE}")
    
    # Main game loop
    frame = 0
    last_score = 0
    
    print(f"Starting game with model: {model_path}")
    start_time = time.time()

    last_pos_x = current_pos_x = pyboy.memory[ADDR_BALL_X]
    last_pos_y = current_pos_y = pyboy.memory[ADDR_BALL_Y]

    stuck = 0
    
    while frame < max_frames:
        # Get the current state
        current_pos_x = pyboy.memory[ADDR_BALL_X]
        current_pos_y = pyboy.memory[ADDR_BALL_Y]
        state = preprocess_state(pyboy, current_pos_x, current_pos_y, last_pos_x, last_pos_y)
        
        # Select action with the model (no exploration)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_values = model(state_tensor)
            action = torch.argmax(action_values).item()
        
        # Apply the action
        
        # Advance game for a few frames
        for _ in range(skip_frames):
            apply_action(pyboy, action)
            pyboy.tick()

        if current_pos_x == last_pos_x and current_pos_y == last_pos_y:
            stuck +=1
        else:
            stuck = 0

        if stuck > 200:
            for _ in range(30):
                pyboy.tick()
            for _ in range(30):
                pyboy.button('a')
                pyboy.tick()
    
        # Track score
        current_score = game_wrapper.score
        
        last_pos_x = current_pos_x
        last_pos_y = current_pos_y

        frame += 1
        
        # Optional: slow down for better visualization
        if render:
            time.sleep(0.01)
    
    # Show final results
    duration = time.time() - start_time
    print(f"\nGame Over!")
    print(f"Final Score: {game_wrapper.score}")
    print(f"Duration: {duration:.2f} seconds, Frames: {frame}")
    
    # Close PyBoy
    pyboy.stop()
    
    return game_wrapper.score


def test_model(model_path, max_frames=20000, skip_frames=4, episodes=25):
    """
    Test trained model
    
    Args:
        model_path: Path to the trained model file
        render: If True, set normal speed for watching, otherwise use max speed
        max_frames: Maximum number of frames to play
        skip_frames: Number of frames to skip between actions
        episodes: Number of episodes
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Initialize PyBoy
    pyboy = PyBoy(ROM_PATH, sound_volume=0)
    pyboy.set_emulation_speed(0)  # Max speed if not watching
    
    # Initialize the game
    game_wrapper = init_game(pyboy)
    
    # Load the model
    model = PinballDQN(STATE_SIZE, ACTION_SIZE)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded model with state size: {STATE_SIZE}")
    
    # Main game loop
    frame = 0
    score = 0
    scores = []
    
    print(f"Starting game with model: {model_path}")
    start_time = time.time()

    last_pos_x = current_pos_x = pyboy.memory[ADDR_BALL_X]
    last_pos_y = current_pos_y = pyboy.memory[ADDR_BALL_Y]
    
    for i in range(episodes):
        init_game(pyboy)

        for _ in range(random.randint(0, 512)):
            pyboy.tick()
        stuck = 0
        for frame in range(max_frames):
            # Get the current state
            current_pos_x = pyboy.memory[ADDR_BALL_X]
            current_pos_y = pyboy.memory[ADDR_BALL_Y]
            state = preprocess_state(pyboy, current_pos_x, current_pos_y, last_pos_x, last_pos_y)
            
            # Select action with the model (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values = model(state_tensor)
                action = torch.argmax(action_values).item()
            
            # Apply the action
            
            # Advance game for a few frames
            for _ in range(skip_frames):
                apply_action(pyboy, action)
                pyboy.tick()
            
            # Track score
            score = game_wrapper.score

            if current_pos_x == last_pos_x and current_pos_y == last_pos_y:
                stuck +=1
            else:
                stuck = 0

            if stuck > 350:
                for _ in range(30):
                    pyboy.tick()
                for _ in range(30):
                    pyboy.button('a')
                    pyboy.tick()
    
            last_pos_x = current_pos_x
            last_pos_y = current_pos_y


            if pyboy.game_wrapper.game_over:
                break

        scores.append(score)
        # Show final results
        duration = time.time() - start_time
        print(f"\nGame {i} Over!")
        print(f"Final Score: {game_wrapper.score:,}")
        print(f"Duration: {duration:.2f} seconds, Frames: {frame}")
    
    print("\Test completed!")
    print(f"Average score: {(sum(scores) / len(scores)):,}")
    print(f"Max score: {max(scores):,}    Min score: {min(scores):,}")

    # Close PyBoy
    pyboy.stop()
    
    return game_wrapper.score