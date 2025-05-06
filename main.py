from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import GameWrapperPokemonPinball

pyboy = PyBoy(
    'rom/Pokemon Pinball (U) [C][!].gbc'
)
pyboy.set_emulation_speed(1)

game_wrapper = pyboy.game_wrapper

# State
pos_x = game_wrapper.ball_x
pos_y = game_wrapper.ball_y

pos_x_velocity = game_wrapper.ball_x_velocity
pos_y_velocity = game_wrapper.ball_y_velocity

# Reward
score = game_wrapper.score

# End game
lives = game_wrapper.balls_left

while pyboy.tick():
    # frame of the game
    pass
    
pyboy.stop()