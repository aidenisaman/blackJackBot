# Blackjack AI Model
A sophisticated reinforcement learning model for playing Blackjack that incorporates card counting and adaptive betting strategies. This implementation combines Q-learning with basic strategy and Hi-Lo card counting system to optimize decision-making and betting patterns.
Features

Reinforcement Learning: Q-learning implementation with dynamic exploration rates
Card Counting: Hi-Lo counting system with true count calculation
Adaptive Betting: Dynamic bet sizing based on count advantage
Basic Strategy: Built-in basic strategy baseline for comparison
Multi-deck Support: Flexible support for 1-8 deck games
Performance Analysis: Comprehensive metrics and visualization tools

Project Structure
Copy├── ai.py           # Core AI implementation with Q-learning and card counting
├── game.py         # Blackjack game engine and card counting logic
└── simulation.py   # Simulation framework for training and evaluation
Requirements

Python 3.7+
NumPy
Matplotlib
pandas
tqdm

Installation

Clone the repository
Install required packages:

bashCopypip install numpy matplotlib pandas tqdm
Usage
Basic Usage
pythonCopyfrom game import BlackjackGame
from ai import BlackjackAI

# Initialize game and AI
game = BlackjackGame(num_decks=6)  # Standard 6-deck game
ai = BlackjackAI()

# Train the model
episodes = 10000
for episode in range(episodes):
    continue_game, reward = game.deal_initial_cards()
    while continue_game:
        state = game.get_game_state()
        action = ai.choose_action(state, training=True)
        continue_game, reward = game.player_action(action)
Running Simulations
pythonCopyfrom simulation import BlackjackSimulator

# Initialize simulator
simulator = BlackjackSimulator()

# Run comparison across different deck sizes
deck_sizes = [1, 2, 4, 6, 8]
results = simulator.run_deck_comparison(
    deck_sizes=deck_sizes,
    episodes_per_size=10000,
    parallel=True
)
Model Components
AI Implementation (ai.py)

Q-learning Parameters:

Learning rate: 0.1
Discount factor: 0.95
Initial epsilon: 0.1
Epsilon decay: 0.99995
Minimum epsilon: 0.01


Betting Strategy:

Minimum bet: 1.0
Maximum bet: 5.0
Count threshold: 2 (for betting decisions)



Game Engine (game.py)

Full implementation of Blackjack rules
Card counting system (Hi-Lo)
Support for multiple decks
Realistic payouts (3:2 for blackjack)

Simulation Framework (simulation.py)

Parallel processing support
Comprehensive performance metrics
Visualization tools
Results persistence

Performance Analysis
The simulation framework provides detailed performance metrics including:

Win/loss rates
Return on Investment (ROI)
Decision distribution
Comparison with basic strategy
Card counting effectiveness
Betting efficiency

Results are automatically saved and visualized in the simulation_results directory.
Model Features

Adaptive Strategy:

Integrates basic strategy with count-based deviations
Dynamically adjusts decisions based on deck composition
Learns optimal deviations through reinforcement


State Representation:

Player hand value
Dealer upcard
Running count
True count
Deck penetration
Usable ace status


Performance Optimization:

Epsilon decay for exploration/exploitation balance
Count-based learning rate adjustment
Deck penetration awareness



Limitations

Does not support splitting pairs
No double down implementation
Limited to hit/stand decisions
Simplified betting spread

Future Improvements

Implement additional player actions:

Splitting pairs
Double down
Insurance


Enhanced betting strategies:

Kelly criterion implementation
Progressive betting systems
Risk management features


Advanced features:

Team play simulation
Shuffle tracking
Multi-spot playing



License
MIT License
