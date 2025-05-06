from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import GameWrapperPokemonPinball
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import time

# Definir la arquitectura de la red neuronal para el agente
class PinballDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(PinballDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Clase para el agente de aprendizaje por refuerzo
class PinballAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Parámetros del aprendizaje
        self.gamma = 0.99    # Factor de descuento
        self.epsilon = 1.0   # Tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Crear redes neurales: principal y objetivo
        self.model = PinballDQN(state_size, action_size)
        self.target_model = PinballDQN(state_size, action_size)
        self.update_target_model()
        
        # Optimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Memoria de experiencia
        self.memory = deque(maxlen=10000)
        
        # Contador para actualización de la red objetivo
        self.update_counter = 0
        
    def update_target_model(self):
        # Copiar pesos de la red principal a la red objetivo
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Política epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Muestrear lote aleatorio de la memoria
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([[t[1]] for t in minibatch])
        rewards = torch.FloatTensor([[t[2]] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([[t[4]] for t in minibatch])
        
        # Q valores actuales
        curr_q_values = self.model(states).gather(1, actions)
        
        # Q valores futuros usando la red objetivo
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
        
        # Calcular valores Q objetivo
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calcular pérdida y actualizar modelo
        loss = F.smooth_l1_loss(curr_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Actualizar red objetivo periódicamente
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.update_target_model()
        
        return loss.item()

# Función para preprocesar el estado del juego
def preprocess_state(game_wrapper):
    state = [
        game_wrapper.ball_x / 255.0,          # Normalizar posición X
        game_wrapper.ball_y / 255.0,          # Normalizar posición Y
        game_wrapper.ball_x_velocity / 10.0,  # Normalizar velocidad X
        game_wrapper.ball_y_velocity / 10.0,  # Normalizar velocidad Y
        game_wrapper.balls_left / 5.0         # Normalizar vidas restantes
    ]
    return state

# Definir acciones posibles
ACTIONS = {
    0: None,                     # No hacer nada
    1: ["left"],                 # Presionar flipper izquierdo
    2: ["right"],                # Presionar flipper derecho
    3: ["left", "right"]         # Presionar ambos flippers
}

# Función para aplicar una acción al juego
def apply_action(pyboy, action_idx):
    
    # Aplicar la acción seleccionada
    if action_idx > 0:
        action = ACTIONS[action_idx]
        if "left" in action:
            pyboy.button('Left')
        if "right" in action:
            pyboy.button('a')

# Función principal de entrenamiento
def train_pinball_agent():
    # Inicializar PyBoy
    pyboy = PyBoy('rom/Pokemon Pinball (U) [C][!].gbc')
    pyboy.set_emulation_speed(0)  # Máxima velocidad
    game_wrapper = pyboy.game_wrapper
    
    # Inicializar agente
    state_size = 5  # [ball_x, ball_y, vel_x, vel_y, lives]
    action_size = len(ACTIONS)
    agent = PinballAgent(state_size, action_size)
    
    # Parámetros de entrenamiento
    num_episodes = 500
    max_frames_per_episode = 10000
    batch_size = 64
    skip_frames = 4  # Saltar frames para acelerar
    
    # Variables para seguimiento
    scores = []
    frame_counts = []
    
    # Ciclo principal de entrenamiento
    pyboy.game_wrapper.start_game()
    for episode in range(num_episodes):
        # Reiniciar juego
        pyboy.game_wrapper.reset_game()
        
        # Esperar a que el juego se estabilice
        for _ in range(50):
            pyboy.tick()
        
        # Obtener estado inicial
        state = preprocess_state(game_wrapper)
        total_reward = 0
        last_score = game_wrapper.score
        
        # Registro de este episodio
        print(f"Episodio {episode+1}/{num_episodes}, Epsilon: {agent.epsilon:.4f}")
        episode_start_time = time.time()
        
        # Ciclo de frames del episodio
        for frame in range(max_frames_per_episode):
            # Seleccionar y ejecutar acción
            action = agent.act(state)
            apply_action(pyboy, action)
            
            # Avanzar el juego varios frames
            for _ in range(skip_frames):
                pyboy.tick()
                if game_wrapper.game_over:
                    break
            
            # Obtener nuevo estado
            next_state = preprocess_state(game_wrapper)
            
            # Calcular recompensa
            current_score = game_wrapper.score
            reward = current_score - last_score  # Recompensa basada en incremento de puntuación
            
            # Añadir pequeña penalización por tiempo sin puntuar
            if reward == 0:
                reward = -0.1
                
            # Verificar si terminó el juego
            done = game_wrapper.game_over
            if done:
                reward = -10  # Penalización por perder
            
            # Almacenar experiencia
            agent.remember(state, action, reward, next_state, done)
            
            # Entrenar la red con lotes de experiencia
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                
            # Actualizar para la siguiente iteración
            state = next_state
            total_reward += reward
            last_score = current_score
            
            # Terminar episodio si el juego terminó
            if done:
                break
        
        # Estadísticas del episodio
        duration = time.time() - episode_start_time
        scores.append(total_reward)
        frame_counts.append(frame)
        
        print(f"  Puntuación: {game_wrapper.score}")
        print(f"  Duración: {duration:.2f} segundos, Frames: {frame}")
        print(f"  Recompensa total: {total_reward:.2f}")
        
        # Guardar modelo cada 50 episodios
        if (episode + 1) % 50 == 0:
            torch.save(agent.model.state_dict(), f"models/pinball_model_ep{episode+1}.pth")
    
    # Guardar modelo final
    torch.save(agent.model.state_dict(), "models/pinball_model_final.pth")
    
    # Cerrar PyBoy
    pyboy.stop()
    
    return scores, frame_counts

if __name__ == "__main__":
    # Ejecutar entrenamiento
    scores, frames = train_pinball_agent()

    # Imprimir resultados finales
    print("\nEntrenamiento completado!")
    print(f"Puntuaciones finales: {scores[-10:]}")
    print(f"Promedio últimas 10 puntuaciones: {sum(scores[-10:]) / 10}")