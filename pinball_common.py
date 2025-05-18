"""
Pokemon Pinball - Common utilities and constants (Optimized)
"""
from pyboy import PyBoy
import numpy as np

# File paths
ROM_PATH = 'rom/Pokemon Pinball (U) [C][!].gbc'
SAVE_STATE_PATH = "rom/Pokemon Pinball Start State.gbc.state"
MODELS_DIR = "models/"

# Define possible actions
ACTIONS = {
    0: None,                     # No action
    1: ["left"],                 # Press left flipper
    2: ["right"],                # Press right flipper
    3: ["shake"]                 # Shake the board
}

# Memory addresses for game state
ADDR_BALL_X = 0xD4B4
ADDR_BALL_Y = 0xD4B6

# Original screen dimensions
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 144

# Reduced screen dimensions (downsampled by 4x)
REDUCED_WIDTH = 40
REDUCED_HEIGHT = 36
IMAGE_SIZE = REDUCED_WIDTH * REDUCED_HEIGHT  # 40x36 = 1440 (much smaller than 23040)

# State dimensions
BASE_STATE_SIZE = 3  # [ball_x, ball_y, lives]
STATE_SIZE = BASE_STATE_SIZE + IMAGE_SIZE  # Adding reduced-size flattened screen image
ACTION_SIZE = len(ACTIONS)

def preprocess_state(pyboy: PyBoy):
    """
    Extract and normalize the game state features, including a downsampled screen image
    """
    # Get basic game state
    base_state = [
        pyboy.memory[ADDR_BALL_X] / 255.0,          # Normalize X position
        pyboy.memory[ADDR_BALL_Y] / 255.0,          # Normalize Y position
        pyboy.game_wrapper.balls_left / 5.0         # Normalize remaining lives
    ]
    
    # Get screen image
    pil_image = pyboy.screen.image
    
    # Convert to grayscale
    gray_img = pil_image.convert('L')
    
    # Downsample to 40x36 (resize first to maintain aspect ratio)
    reduced_img = gray_img.resize((REDUCED_WIDTH, REDUCED_HEIGHT))
    
    # Convert to numpy array and normalize to 0-1
    img_array = np.array(reduced_img).astype(np.float32) / 255.0
    
    # Flatten the image array
    flat_img = img_array.flatten()
    
    # Combine base state with image data
    full_state = base_state + flat_img.tolist()
    
    return full_state

def apply_action(pyboy, action_idx):
    """
    Apply the selected action to the game
    """
    if action_idx > 0:
        action = ACTIONS[action_idx]
        if "left" in action:
            pyboy.button('Left')
        if "right" in action:
            pyboy.button('a')
        if "shake" in action:
            pyboy.button('b')

def init_game(pyboy: PyBoy, frames_to_stabilize=50):
    """
    Initialize the game from the save state and wait for it to stabilize
    """
    with open(SAVE_STATE_PATH, "rb") as f:
        pyboy.load_state(f)
    
    # Wait for the game to stabilize
    for _ in range(frames_to_stabilize):
        pyboy.tick()
    
    return pyboy.game_wrapper