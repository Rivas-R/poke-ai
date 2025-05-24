from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import GameWrapperPokemonPinball
import time
from PIL import Image
import numpy as np


REDUCED_WIDTH = 40
REDUCED_HEIGHT = 36

import torch

"""# Check if CUDA (NVIDIA GPU) is available
print("CUDA available:", torch.cuda.is_available())

# List available GPUs
print("GPUs:", torch.cuda.device_count())

# Current GPU name
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))"""

def play():
    pyboy = PyBoy('rom/Pokemon Pinball (U) [C][!].gbc')
    pyboy.set_emulation_speed(1)

    game_wrapper = pyboy.game_wrapper

    # State
    ADDR_BALL_X = 0xD4B4
    ADDR_BALL_Y = 0xD4B6

    ADDR_BALL_X_VELOCITY = 0xD4BB
    ADDR_BALL_X_VELOCITY_2 = 0xD4BC
    ADDR_BALL_Y_VELOCITY = 0xD4BD
    ADDR_BALL_Y_VELOCITY_2 = 0xD4BE

    
    # Reward
    score = game_wrapper.score

    # End game
    lives = game_wrapper.balls_left

    iframes = 0
    interval = 100

    with open("rom/Pokemon Pinball Start State.gbc.state", "rb") as f:
        pyboy.load_state(f)
        
    stuck = stuck_initial = 10

    last_pos_x = current_pos_x = pyboy.memory[ADDR_BALL_X] / 168
    last_pos_y = current_pos_y = pyboy.memory[ADDR_BALL_Y] /168

    while pyboy.tick():
        if iframes % interval == 0:
            #screenshot(pyboy, f'screenshots/screenshot_{iframes//interval}.png')
            pass

        #print(pyboy.game_wrapper.pokemon_caught_in_session)
        #time.sleep(0.4)
        current_pos_x = pyboy.memory[ADDR_BALL_X] / 168
        current_pos_y = pyboy.memory[ADDR_BALL_Y] / 168

        #print(f'pos_x: {pos_x / 168.0}  pos_y:{pos_y / 168.0}')
        #print(pyboy.game_wrapper.balls_left)

        if abs(current_pos_x - last_pos_x) < 0.015 and abs(current_pos_y - last_pos_y) < 0.015:
                #reward -= stuck # Small penalty for doing nothing
                stuck *= 1.005
        else:
            stuck = stuck_initial
        
        print(stuck)
    
        last_pos_x = current_pos_x
        last_pos_y = current_pos_y
        #time.sleep(.4)
        
        iframes += 1
        #print(iframes % interval)

    pyboy.stop()


def screenshot(pyboy, img_name:str):
    # Get screen image
    pil_image = pyboy.screen.image
    
    # Convert to grayscale
    gray_img = pil_image.convert('L')
    
    # Downsample to 40x36 (resize first to maintain aspect ratio)
    reduced_img = gray_img.resize((REDUCED_WIDTH, REDUCED_HEIGHT))
    
    # Convert to binary (black and white)
    #bw_img = reduced_img.point(lambda x: 0 if x < 128 else 255, '1')

    reduced_img.save(img_name)
    pass

if __name__ == "__main__":
    play()

    pass