"""
Pokemon Pinball - Main menu to select between training and playing (Optimized)
"""
import os
import sys
import importlib

# Import common first to set up constants
from pinball_common import MODELS_DIR, STATE_SIZE, IMAGE_SIZE, BASE_STATE_SIZE, REDUCED_WIDTH, REDUCED_HEIGHT

# Then import the other modules
from pinball_training import train_pinball_agent
from pinball_play import play_pinball_with_model

def list_available_models():
    """
    List all available trained models
    """
    if not os.path.exists(MODELS_DIR):
        print(f"No models directory found. Please train a model first.")
        return []
    
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    
    if not models:
        print("No trained models found. Please train a model first.")
        return []
    
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    return models

def main_menu():
    """
    Display main menu and handle user input
    """
    while True:
        print("\n" + "="*50)
        print("POKEMON PINBALL AI - MAIN MENU (OPTIMIZED)")
        print("="*50)
        print("1. Train new model")
        print("2. Play with trained model")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            train_menu()
        elif choice == '2':
            play_menu()
        elif choice == '3':
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

def train_menu():
    """
    Menu for training options
    """
    print("\n" + "="*50)
    print("TRAINING OPTIONS")
    print("="*50)
    
    try:
        episodes = int(input("Number of episodes (default: 500): ") or "500")
        batch_size = int(input("Batch size (default: 64): ") or "64")
        max_frames = int(input("Max frames per episode (default: 10000): ") or "10000")
        skip_frames = int(input("Skip frames (default: 6): ") or "6")
        
        print("\nStarting training with the following parameters:")
        print(f"Episodes: {episodes}")
        print(f"Batch Size: {batch_size}")
        print(f"Max Frames: {max_frames}")
        print(f"Skip Frames: {skip_frames}")
        print(f"State Size: {STATE_SIZE} ({REDUCED_WIDTH}x{REDUCED_HEIGHT} visual input)")
        
        # Confirm start training
        confirm = input("\nStart training? (y/n): ").lower()
        if confirm == 'y':            
            train_pinball_agent(
                num_episodes=episodes,
                batch_size=batch_size,
                max_frames_per_episode=max_frames,
                skip_frames=skip_frames
            )
    
    except ValueError:
        print("Invalid input. Please enter numeric values.")

def play_menu():
    """
    Menu for playing with a trained model
    """
    models = list_available_models()
    
    if not models:
        return
    
    try:
        model_idx = int(input("\nSelect model number to play with: ")) - 1
        
        if model_idx < 0 or model_idx >= len(models):
            print("Invalid model number.")
            return
        
        model_path = os.path.join(MODELS_DIR, models[model_idx])
        
        render = input("Watch the game? (y/n, default: y): ").lower() != 'n'
        max_frames = int(input("Max frames to play (default: 20000): ") or "20000")
        skip_frames = int(input("Skip frames between actions (default: 2): ") or "2")
        
        print(f"\nPlaying with model: {models[model_idx]}")
        print(f"Render: {'Yes' if render else 'No'}")
        print(f"Max Frames: {max_frames}")
        print(f"Skip Frames: {skip_frames}")
        print(f"State Size: {STATE_SIZE} (includes {REDUCED_WIDTH}x{REDUCED_HEIGHT} visual input)")
        
        # Start playing
        play_pinball_with_model(
            model_path=model_path,
            render=render,
            max_frames=max_frames,
            skip_frames=skip_frames
        )
    
    except ValueError:
        print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Display main menu
    main_menu()