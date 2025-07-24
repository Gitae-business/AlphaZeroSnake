import torch
import numpy as np
from game.board import Board
from utils.mcts import MCTS, Node

N_PLAYOUT = 40

class SelfPlay:
    def __init__(self, board_width, board_height, num_actions, num_snakes=1, model=None):
        self.board = Board(board_width, board_height, num_snakes) # num_snakes is now 1
        self.input_shape = (3, board_width, board_height) # Channels, Width, Height
        self.num_actions = num_actions
        self.num_snakes = num_snakes # Should be 1

        self.model = model
        if self.model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode

        # Only one MCTS agent for a single snake
        self.mcts_agent = MCTS(lambda board, snake_id: self._policy_value_fn(board, snake_id), 
                                 self.num_actions, self.num_snakes, n_playout=N_PLAYOUT)

    def _policy_value_fn(self, board, snake_id):
        state_tensor = torch.tensor(board.get_state(snake_id), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_ps, value = self.model(state_tensor)
            
        action_probs = np.exp(log_ps.cpu().numpy()[0])

        # Mask invalid moves and re-normalize probabilities
        valid_moves = board.get_valid_moves(snake_id)
        masked_action_probs = np.zeros_like(action_probs)
        for move_idx in valid_moves:
            masked_action_probs[move_idx] = action_probs[move_idx]
        
        # Re-normalize
        sum_masked_probs = np.sum(masked_action_probs)
        if sum_masked_probs > 0:
            masked_action_probs /= sum_masked_probs
        else:
            # If no valid moves (e.g., trapped), distribute probability equally among all moves
            # This case should ideally be handled by game over logic before reaching here
            masked_action_probs = np.ones_like(action_probs) / len(action_probs)

        return masked_action_probs, value.item()

    def play_game(self, temp=1.0):
        self.board.initialize()
        self.mcts_agent.root = Node(None, 1.0) # Reset MCTS tree for new game
        
        game_history = []
        current_player_idx = 0 # Always 0 for a single snake

        while not self.board.is_game_over:
            # For single snake, just check if the snake is alive
            if not self.board.snakes[current_player_idx].is_alive:
                break # Game over if the only snake is dead

            state = self.board.get_state(current_player_idx)
            acts, probs = self.mcts_agent.get_move_probs(self.board, temp=temp, current_player=current_player_idx)
            
            # Store (state, policy, current_player_id) for this turn
            game_history.append([state, probs, current_player_idx, None]) # state, policy, player_id, value

            action = np.random.choice(acts, p=probs)
            self.board.update(current_player_idx, action)
            
            # Only update the MCTS tree for the current player based on the actual move made
            self.mcts_agent.update_with_move(action)

        # Assign game result to the value of each state in the history
        # For single snake, the value is the final length achieved
        final_snake_length = self.board.snakes[current_player_idx].length # Assuming snake object has a 'length' attribute

        # If the snake died, we might want to penalize it, or just use the length it achieved
        # For simplicity, let's use the length as the value.
        # If you want to penalize death more, you could make it negative or 0 for death.
        game_result = final_snake_length

        for i in reversed(range(len(game_history))):
            game_history[i][3] = game_result # Assign the final game result to all states

        # Filter out turns where the player was already dead or game ended abruptly
        # And format for saving
        formatted_history = []
        for state, policy, player_id, value in game_history:
            formatted_history.append({"state": state, "policy": policy, "value": value})

        return formatted_history
