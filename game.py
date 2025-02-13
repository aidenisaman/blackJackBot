# File: game.py
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"

class Action(Enum):
    HIT = "hit"
    STAND = "stand"

@dataclass
class Card:
    suit: Suit
    rank: str
    value: int

class CardCounter:
    def __init__(self):
        # Hi-Lo count values: 2-6 = +1, 7-9 = 0, 10-A = -1
        self.count_values = {
            '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
            '7': 0, '8': 0, '9': 0,
            '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
        }
        self.running_count = 0
        self.cards_seen = 0
    
    def update_count(self, card: Card):
        """Update running count based on seen card"""
        self.running_count += self.count_values[card.rank]
        self.cards_seen += 1
    
    def get_true_count(self, decks_remaining: float) -> float:
        """Convert running count to true count"""
        if decks_remaining <= 0:
            return 0
        return self.running_count / decks_remaining
    
    def reset(self):
        """Reset counter"""
        self.running_count = 0
        self.cards_seen = 0

class Deck:
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self._create_deck()
        self.shuffle()
    
    def _create_deck(self):
        ranks = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10,
                'J':10, 'Q':10, 'K':10, 'A':11}
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank, value in ranks.items():
                    self.cards.append(Card(suit, rank, value))
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw(self) -> Optional[Card]:
        return self.cards.pop() if self.cards else None
    
    def cards_remaining(self) -> int:
        return len(self.cards)

class Hand:
    def __init__(self):
        self.cards: List[Card] = []
    
    def add_card(self, card: Card):
        self.cards.append(card)
    
    def get_value(self) -> int:
        value = 0
        aces = 0
        
        for card in self.cards:
            if card.rank == 'A':
                aces += 1
            value += card.value
        
        # Adjust for aces
        while value > 21 and aces:
            value -= 10
            aces -= 1
        
        return value
    
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.get_value() == 21
    
    def clear(self):
        self.cards = []

class BlackjackGame:
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        self.deck = Deck(num_decks)
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.game_over = False
        self.player_wins = 0
        self.dealer_wins = 0
        self.ties = 0
        self.shuffle_threshold = 52 * num_decks // 4
        self.current_bet = 1.0
        self.counter = CardCounter()
    
    def draw_card(self, hand: Hand) -> Optional[Card]:
        """Draw card and update count"""
        card = self.deck.draw()
        if card:
            hand.add_card(card)
            self.counter.update_count(card)
        return card
    
    def deal_initial_cards(self) -> Tuple[bool, float]:
        """Deal initial cards and return if game should continue and initial reward"""
        self.player_hand.clear()
        self.dealer_hand.clear()
        self.game_over = False
        
        # Check if we need to reshuffle
        if self.deck.cards_remaining() < self.shuffle_threshold:
            self.deck = Deck(self.num_decks)
            self.counter.reset()
        
        # Deal two cards each
        self.draw_card(self.player_hand)
        self.draw_card(self.dealer_hand)
        self.draw_card(self.player_hand)
        self.draw_card(self.dealer_hand)
        
        # Check for initial blackjack
        if self.player_hand.is_blackjack() or self.dealer_hand.is_blackjack():
            self.game_over = True
            reward = self.get_reward(self.player_hand, self.dealer_hand)
            if reward > 0:
                self.player_wins += 1
            elif reward < 0:
                self.dealer_wins += 1
            else:
                self.ties += 1
            return False, reward
        
        return True, 0
    
    def get_reward(self, player_hand: Hand, dealer_hand: Hand) -> float:
        """Calculate reward with realistic payouts"""
        player_value = player_hand.get_value()
        dealer_value = dealer_hand.get_value()
        
        # Player busts
        if player_value > 21:
            return -self.current_bet
        
        # Dealer busts
        if dealer_value > 21:
            return self.current_bet
        
        # Player blackjack
        if player_hand.is_blackjack():
            if dealer_hand.is_blackjack():
                return 0  # Push on both blackjack
            return self.current_bet * 1.5  # Blackjack pays 3:2
        
        # Dealer blackjack
        if dealer_hand.is_blackjack():
            return -self.current_bet
        
        # Regular win/loss/push
        if player_value > dealer_value:
            return self.current_bet
        elif player_value < dealer_value:
            return -self.current_bet
        return 0  # Push
    
    def player_action(self, action: Action) -> Tuple[bool, float]:
        """Execute player action and return continue flag and reward"""
        if self.game_over:
            return False, 0
        
        reward = 0
        if action == Action.HIT:
            card = self.draw_card(self.player_hand)
            if not card:
                self.deck = Deck(self.num_decks)
                self.counter.reset()
                card = self.draw_card(self.player_hand)
            
            if self.player_hand.get_value() > 21:
                self.game_over = True
                reward = self.get_reward(self.player_hand, self.dealer_hand)
                self.dealer_wins += 1
                return False, reward
        elif action == Action.STAND:
            self._dealer_play()
            self.game_over = True
            reward = self.get_reward(self.player_hand, self.dealer_hand)
            if reward > 0:
                self.player_wins += 1
            elif reward < 0:
                self.dealer_wins += 1
            else:
                self.ties += 1
            return False, reward
        
        return True, reward
    
    def _dealer_play(self):
        """Execute dealer's play according to house rules"""
        while self.dealer_hand.get_value() < 17:
            card = self.draw_card(self.dealer_hand)
            if not card:
                self.deck = Deck(self.num_decks)
                self.counter.reset()
                card = self.draw_card(self.dealer_hand)
    
    def get_game_state(self) -> dict:
        """Return current game state for AI training"""
        decks_remaining = self.deck.cards_remaining() / 52
        return {
            'player_hand': [(c.rank, c.suit.value) for c in self.player_hand.cards],
            'player_value': self.player_hand.get_value(),
            'dealer_visible_card': (self.dealer_hand.cards[0].rank, 
                                  self.dealer_hand.cards[0].suit.value) if self.dealer_hand.cards else None,
            'dealer_value': self.dealer_hand.get_value(),
            'cards_remaining': self.deck.cards_remaining(),
            'deck_penetration': self.deck.cards_remaining() / (52 * self.num_decks),
            'game_over': self.game_over,
            'player_wins': self.player_wins,
            'dealer_wins': self.dealer_wins,
            'ties': self.ties,
            'has_blackjack': self.player_hand.is_blackjack(),
            'num_decks': self.num_decks,
            'running_count': self.counter.running_count,
            'true_count': self.counter.get_true_count(decks_remaining),
            'cards_seen': self.counter.cards_seen
        }

def play_example_game():
    """Example of how to play a manual game"""
    game = BlackjackGame()
    continue_game, reward = game.deal_initial_cards()
    
    if not continue_game:
        print(f"Game ended immediately with reward: {reward}")
        return
    
    while not game.game_over:
        state = game.get_game_state()
        print(f"\nPlayer hand: {state['player_hand']} (Value: {state['player_value']})")
        print(f"Dealer shows: {state['dealer_visible_card']}")
        print(f"Running Count: {state['running_count']}")
        print(f"True Count: {state['true_count']:.2f}")
        
        while True:
            action = input("Enter action (hit/stand): ").lower()
            if action in ['hit', 'stand']:
                break
            print("Invalid action. Please enter 'hit' or 'stand'.")
        
        continue_game, reward = game.player_action(Action.HIT if action == 'hit' else Action.STAND)
        if not continue_game:
            print(f"Reward: {reward}")
    
    final_state = game.get_game_state()
    print("\nGame Over!")
    print(f"Final player hand: {final_state['player_hand']} (Value: {final_state['player_value']})")
    print(f"Final dealer hand value: {final_state['dealer_value']}")
    print(f"Final Running Count: {final_state['running_count']}")
    print(f"Final True Count: {final_state['true_count']:.2f}")
    print(f"Results - Wins: {final_state['player_wins']}, "
          f"Losses: {final_state['dealer_wins']}, "
          f"Ties: {final_state['ties']}")

if __name__ == "__main__":
    play_example_game()