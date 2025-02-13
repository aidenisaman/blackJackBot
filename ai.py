from typing import Tuple, Dict, List
import numpy as np
from collections import defaultdict
import pickle
import json
import os
from datetime import datetime

class BlackjackAI:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        # Q-learning parameters
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99995
        
        # Betting parameters
        self.min_bet = 1.0
        self.max_bet = 5.0
        self.count_threshold = 2  # True count threshold for betting decisions
        
        # Initialize tracking metrics
        self.state_action_counts = defaultdict(lambda: defaultdict(int))
        self.training_history = {
            'win_rates': [],
            'episode_rewards': [],
            'decisions': defaultdict(int),
            'total_return': 0.0,
            'blackjack_frequency': 0,
            'bust_frequency': 0,
            'avg_bet_size': [],
            'count_based_decisions': defaultdict(int)
        }
        
        # Strategy statistics
        self.deck_influence = defaultdict(lambda: defaultdict(float))
    
    def get_state_key(self, game_state: dict) -> Tuple:
        """Convert game state to a hashable tuple for Q-table lookup"""
        player_value = game_state['player_value']
        dealer_card = game_state['dealer_visible_card'][0] if game_state['dealer_visible_card'] else '0'
        
        # Detect usable ace and create simple hand composition
        has_usable_ace = self._has_usable_ace(game_state['player_hand'])
        
        # Get count information
        true_count = game_state['true_count']
        count_state = 'high' if true_count >= 2 else 'low' if true_count <= -2 else 'neutral'
        
        # Get deck penetration state
        deck_penetration = game_state['deck_penetration']
        deck_state = 'low' if deck_penetration < 0.25 else 'medium' if deck_penetration < 0.5 else 'high'
        
        return (player_value, dealer_card, has_usable_ace, count_state, deck_state)
    
    def _has_usable_ace(self, hand: List[Tuple]) -> bool:
        """Determine if hand has a usable ace (counted as 11)"""
        ace_count = sum(1 for card in hand if card[0] == 'A')
        if not ace_count:
            return False
            
        # Calculate hand value counting aces as 1
        non_ace_value = sum(self._card_value(card[0]) for card in hand if card[0] != 'A')
        
        # Check if any ace can be counted as 11
        return non_ace_value + 11 + (ace_count - 1) <= 21
    
    def _card_value(self, rank: str) -> int:
        """Get numerical value of a card rank"""
        if rank in ['K', 'Q', 'J']:
            return 10
        elif rank == 'A':
            return 1
        return int(rank) if rank.isdigit() else 0
    
    def get_bet_size(self, game_state: dict) -> float:
        """Determine bet size based on card counting"""
        true_count = game_state['true_count']
        deck_penetration = game_state['deck_penetration']
        
        # Increase bet with positive count and deep penetration
        if true_count > 0 and deck_penetration < 0.75:
            # Bet size increases with true count
            bet_multiplier = min(max(1, true_count / self.count_threshold), 
                               self.max_bet / self.min_bet)
            return self.min_bet * bet_multiplier
        
        return self.min_bet
    
    def should_deviate_from_basic(self, game_state: dict) -> bool:
        """Determine if we should deviate from basic strategy based on count"""
        true_count = game_state['true_count']
        player_value = game_state['player_value']
        dealer_card = game_state['dealer_visible_card'][0] if game_state['dealer_visible_card'] else '0'
        
        # Key deviations based on true count
        if true_count >= 3:  # Rich deck (lots of high cards)
            if player_value == 16 and dealer_card == '10':  # Stand on 16 vs 10
                return True
            if player_value == 15 and dealer_card == '10':  # Stand on 15 vs 10
                return True
        elif true_count <= -2:  # Poor deck (lots of low cards)
            if player_value >= 12 and player_value <= 16:  # Hit hard 12-16 vs dealer 2-6
                if dealer_card in ['2', '3', '4', '5', '6']:
                    return True
        
        return False
    
    def choose_action(self, game_state: dict, training: bool = True) -> str:
        """Choose action using epsilon-greedy strategy with counting influence"""
        state = self.get_state_key(game_state)
        
        # Decay epsilon during training
        if training:
            self.epsilon = max(self.min_epsilon, 
                             self.epsilon * self.epsilon_decay)
        
        # Check for count-based deviations if not training
        if not training and self.should_deviate_from_basic(game_state):
            action = self._get_deviation_action(game_state)
            self.training_history['count_based_decisions'][action] += 1
        else:
            # Standard exploration/exploitation
            if training and np.random.random() < self.epsilon:
                action = np.random.choice(['hit', 'stand'])
            else:
                hit_value = self.q_table[state]['hit']
                stand_value = self.q_table[state]['stand']
                action = 'hit' if hit_value >= stand_value else 'stand'
        
        # Track decision
        self.state_action_counts[state][action] += 1
        self.training_history['decisions'][action] += 1
        
        return action
    
    def _get_deviation_action(self, game_state: dict) -> str:
        """Get action for count-based strategy deviation"""
        player_value = game_state['player_value']
        dealer_card = game_state['dealer_visible_card'][0]
        true_count = game_state['true_count']
        
        # Conservative decision matrix based on true count
        if true_count >= 3:
            if player_value in [15, 16] and dealer_card in ['10', 'J', 'Q', 'K']:
                return 'stand'
        elif true_count <= -2:
            if player_value in [12, 13, 14, 15, 16] and dealer_card in ['2', '3', '4', '5', '6']:
                return 'hit'
        
        # Default to basic strategy if no deviation applies
        return self._get_basic_strategy_action(game_state)
    
    def _get_basic_strategy_action(self, game_state: dict) -> str:
        """Get action based on basic strategy"""
        player_value = game_state['player_value']
        dealer_card = game_state['dealer_visible_card'][0]
        has_usable_ace = self._has_usable_ace(game_state['player_hand'])
        
        if has_usable_ace:  # Soft hands
            if player_value <= 17:
                return 'hit'
            return 'stand'
        else:  # Hard hands
            dealer_value = 10 if dealer_card in ['10','J','Q','K'] else \
                         11 if dealer_card == 'A' else int(dealer_card)
            
            if player_value <= 11:
                return 'hit'
            elif player_value >= 17:
                return 'stand'
            elif player_value >= 12:  # 12-16
                if dealer_value >= 7:
                    return 'hit'
                return 'stand'
        
        return 'hit'  # Default action
    
    def update(self, game_state: dict, action: str, reward: float, next_state: dict):
        """Update Q-values and tracking metrics"""
        state = self.get_state_key(game_state)
        next_state_key = self.get_state_key(next_state)
        
        # Track metrics
        self.training_history['total_return'] += reward
        self.training_history['episode_rewards'].append(reward)
        
        if game_state.get('has_blackjack', False):
            self.training_history['blackjack_frequency'] += 1
        if game_state.get('player_value', 0) > 21:
            self.training_history['bust_frequency'] += 1
        
        # Track bet sizes
        current_bet = self.get_bet_size(game_state)
        self.training_history['avg_bet_size'].append(current_bet)
        
        # Get max Q-value for next state
        next_max_q = max(
            self.q_table[next_state_key]['hit'],
            self.q_table[next_state_key]['stand']
        )
        
        # Q-learning update with count influence
        current_q = self.q_table[state][action]
        count_factor = abs(game_state['true_count']) / 10  # Count influence factor
        
        # Adjust learning rate based on count confidence
        adjusted_lr = self.learning_rate * (1 + count_factor)
        
        self.q_table[state][action] = current_q + adjusted_lr * (
            reward + self.discount_factor * next_max_q - current_q
        )
    
    def analyze_performance(self) -> Dict:
        """Generate comprehensive performance analysis"""
        total_episodes = len(self.training_history['episode_rewards'])
        if total_episodes == 0:
            return {'error': 'No training data available'}
        
        total_decisions = sum(self.training_history['decisions'].values())
        
        return {
            'overall_stats': {
                'total_episodes': total_episodes,
                'average_reward': np.mean(self.training_history['episode_rewards']),
                'total_return': self.training_history['total_return'],
                'blackjack_rate': self.training_history['blackjack_frequency'] / max(1, total_episodes),
                'bust_rate': self.training_history['bust_frequency'] / max(1, total_episodes),
                'win_rate': len([r for r in self.training_history['episode_rewards'] if r > 0]) / max(1, total_episodes),
                'average_bet_size': np.mean(self.training_history['avg_bet_size'])
            },
            'strategy_stats': {
                'hit_frequency': self.training_history['decisions']['hit'] / max(1, total_decisions),
                'stand_frequency': self.training_history['decisions']['stand'] / max(1, total_decisions),
                'count_based_deviations': dict(self.training_history['count_based_decisions'])
            }
        }
    
    def save_model(self, filename: str):
        """Save the model and training history"""
        model_data = {
            'q_table': dict(self.q_table),
            'training_history': self.training_history,
            'state_action_counts': dict(self.state_action_counts),
            'config': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'min_bet': self.min_bet,
                'max_bet': self.max_bet,
                'count_threshold': self.count_threshold
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename: str):
        """Load a saved model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.training_history = model_data['training_history']
        self.state_action_counts = defaultdict(lambda: defaultdict(int), 
                                            model_data.get('state_action_counts', {}))
        
        config = model_data.get('config', {})
        self.learning_rate = config.get('learning_rate', self.learning_rate)
        self.discount_factor = config.get('discount_factor', self.discount_factor)
        self.epsilon = config.get('epsilon', self.epsilon)
        self.min_bet = config.get('min_bet', self.min_bet)
        self.max_bet = config.get('max_bet', self.max_bet)
        self.count_threshold = config.get('count_threshold', self.count_threshold)

def create_baseline_strategy() -> BlackjackAI:
    """Create an AI with basic strategy rules"""
    ai = BlackjackAI(epsilon=0)  # No exploration for baseline
    
    # Set up the basic strategy table
    for player_value in range(4, 22):
        for dealer_card in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']:
            for has_ace in [True, False]:
                for count_state in ['high', 'neutral', 'low']:
                    for deck_state in ['high', 'medium', 'low']:
                        state = (player_value, dealer_card, has_ace, count_state, deck_state)
                        
                        if has_ace:  # Soft hands
                            if player_value <= 17:
                                ai.q_table[state]['hit'] = 1
                                ai.q_table[state]['stand'] = -1
                            else:
                                ai.q_table[state]['hit'] = -1
                                ai.q_table[state]['stand'] = 1
                        else:  # Hard hands
                            dealer_value = 10 if dealer_card in ['10','J','Q','K'] else \
                                         11 if dealer_card == 'A' else int(dealer_card)
                            
                            if player_value <= 11:
                                ai.q_table[state]['hit'] = 1
                                ai.q_table[state]['stand'] = -1
                            elif player_value >= 17:
                                ai.q_table[state]['hit'] = -1
                                ai.q_table[state]['stand'] = 1
                            else:  # 12-16
                                if dealer_value >= 7:
                                    ai.q_table[state]['hit'] = 1
                                    ai.q_table[state]['stand'] = -1
                                else:
                                    ai.q_table[state]['hit'] = -1
                                    ai.q_table[state]['stand'] = 1
    
    return ai

if __name__ == "__main__":
    # Example usage
    ai = BlackjackAI()
    print("AI initialized with epsilon:", ai.epsilon)
    
    # Create and display basic strategy
    baseline = create_baseline_strategy()
    print("Baseline strategy created")