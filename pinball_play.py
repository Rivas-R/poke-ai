from pyboy import PyBoy
import torch
import time

from pinball_training import PinballDQN,  preprocess_state, apply_action, ACTIONS

def play_pinball_with_model(model_path, render=True, max_frames=20000):
    # Initialize PyBoy
    pyboy = PyBoy('rom/Pokemon Pinball (U) [C][!].gbc')
    
    if render:
        pyboy.set_emulation_speed(1)  # Set normal speed for watching
    else:
        pyboy.set_emulation_speed(0)  # Max speed if not watching
    
    game_wrapper = pyboy.game_wrapper
    
    # Load the model
    state_size = 5  # [ball_x, ball_y, vel_x, vel_y, lives]
    action_size = len(ACTIONS)
    model = PinballDQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    # Wait for the game to stabilize
    for _ in range(50):
        pyboy.tick()
    
    # Main game loop
    frame = 0
    last_score = 0
    skip_frames = 2  # Fewer skip frames for smoother gameplay
    
    print(f"Starting game with model: {model_path}")
    start_time = time.time()
    
    while not game_wrapper.game_over and frame < max_frames:
        # Get the current state
        state = preprocess_state(game_wrapper)
        
        # Select action with the model (no exploration)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_values = model(state_tensor)
            action = torch.argmax(action_values).item()
        
        # Apply the action
        apply_action(pyboy, action)
        
        # Advance game for a few frames
        for _ in range(skip_frames):
            pyboy.tick()
            if game_wrapper.game_over:
                break
        
        # Track score
        current_score = game_wrapper.score
        if current_score > last_score:
            print(f"Score: {current_score}")
            last_score = current_score
        
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

if __name__ == "__main__":
    # Replace with your model path
    model_path = "models/pinball_model_ep250.pth"  # or any other saved model
    
    # Set render=True to watch the agent play
    play_pinball_with_model(model_path, render=True)