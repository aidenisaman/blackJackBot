import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pandas as pd
from datetime import datetime
import json
import os
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from game import BlackjackGame, Action
from ai import BlackjackAI, create_baseline_strategy

class BlackjackSimulator:
    def __init__(self, save_dir: str = "simulation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_deck_comparison(self, 
                          deck_sizes: List[int], 
                          episodes_per_size: int = 10000,
                          parallel: bool = True) -> Dict:
        """Run simulations with different numbers of decks"""
        if parallel and len(deck_sizes) > 1:
            return self._run_parallel_comparison(deck_sizes, episodes_per_size)
        
        results = {}
        for num_decks in deck_sizes:
            print(f"\nTraining with {num_decks} deck(s)...")
            result = self._train_and_evaluate(num_decks, episodes_per_size)
            results[num_decks] = result
        
        self._save_comparison_results(results)
        self._plot_comparison_results(results)
        
        return results
    
    def _train_and_evaluate(self, num_decks: int, episodes: int) -> Dict:
        """Train AI and evaluate performance for a specific deck size"""
        game = BlackjackGame(num_decks=num_decks)
        ai = BlackjackAI()
        
        # Training phase
        training_stats = self._train_ai(ai, game, episodes)
        
        # Evaluation phase
        eval_stats = self._evaluate_ai(ai, game)
        
        # Compare with baseline strategy
        baseline_ai = create_baseline_strategy()
        baseline_stats = self._evaluate_ai(baseline_ai, game)
        
        result = {
            'num_decks': num_decks,
            'training_stats': training_stats,
            'evaluation_stats': eval_stats,
            'baseline_stats': baseline_stats,
            'ai_analysis': ai.analyze_performance()
        }
        
        # Save individual model
        self._save_model(ai, num_decks)
        
        return result
    
    def _train_ai(self, ai: BlackjackAI, game: BlackjackGame, episodes: int) -> Dict:
        """Train AI and collect training statistics"""
        wins = losses = ties = total_games = 0
        rewards = []
        win_rates = []
        
        for episode in tqdm(range(episodes), desc="Training"):
            continue_game, reward = game.deal_initial_cards()
            total_games += 1
            
            if not continue_game:  # Handle initial blackjack
                rewards.append(reward)
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    ties += 1
                continue
            
            episode_rewards = [reward]  # Include initial reward
            
            while continue_game:
                current_state = game.get_game_state()
                action = ai.choose_action(current_state, training=True)
                continue_game, reward = game.player_action(
                    Action.HIT if action == 'hit' else Action.STAND
                )
                
                new_state = game.get_game_state()
                episode_rewards.append(reward)
                
                # Update AI
                ai.update(current_state, action, reward, new_state)
                
                if not continue_game:
                    if reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    else:
                        ties += 1
            
            rewards.extend(episode_rewards)
            
            # Track win rate every 100 episodes
            if (episode + 1) % 100 == 0:
                win_rate = wins / max(1, total_games)  # Avoid division by zero
                win_rates.append(win_rate)
                wins = losses = ties = total_games = 0
        
        return {
            'final_win_rate': win_rates[-1] if win_rates else 0,
            'win_rate_history': win_rates,
            'reward_history': rewards,
            'average_reward': np.mean(rewards) if rewards else 0
        }
    
    def _evaluate_ai(self, ai: BlackjackAI, game: BlackjackGame, 
                    episodes: int = 1000) -> Dict:
        """Evaluate AI performance"""
        wins = losses = ties = total_games = 0
        bet_return = 0
        decisions = {'hit': 0, 'stand': 0}
        rewards = []
        
        for _ in tqdm(range(episodes), desc="Evaluating"):
            continue_game, reward = game.deal_initial_cards()
            total_games += 1
            rewards.append(reward)
            
            while continue_game:
                state = game.get_game_state()
                action = ai.choose_action(state, training=False)
                decisions[action] += 1
                
                continue_game, reward = game.player_action(
                    Action.HIT if action == 'hit' else Action.STAND
                )
                rewards.append(reward)
            
            bet_return += reward
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                ties += 1
        
        total_games = max(1, total_games)  # Avoid division by zero
        total_decisions = max(1, sum(decisions.values()))  # Avoid division by zero
        
        return {
            'win_rate': wins / total_games,
            'loss_rate': losses / total_games,
            'tie_rate': ties / total_games,
            'bet_return': bet_return,
            'roi': (bet_return / total_games) * 100,
            'average_reward': np.mean(rewards) if rewards else 0,
            'decisions': {
                'hit_ratio': decisions['hit'] / total_decisions,
                'stand_ratio': decisions['stand'] / total_decisions,
                'total_decisions': sum(decisions.values())
            }
        }
    
    def _run_parallel_comparison(self, deck_sizes: List[int], 
                               episodes_per_size: int) -> Dict:
        """Run deck comparisons in parallel"""
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._train_and_evaluate, num_decks, episodes_per_size)
                for num_decks in deck_sizes
            ]
            results = {
                num_decks: future.result()
                for num_decks, future in zip(deck_sizes, futures)
            }
        
        self._save_comparison_results(results)
        self._plot_comparison_results(results)
        
        return results
    
    def _save_model(self, ai: BlackjackAI, num_decks: int):
        """Save trained model and its analysis"""
        filename = os.path.join(
            self.save_dir, 
            f"blackjack_model_{num_decks}deck_{self.timestamp}"
        )
        ai.save_model(filename)
    
    def _save_comparison_results(self, results: Dict):
        """Save comparison results to JSON"""
        filename = os.path.join(
            self.save_dir,
            f"comparison_results_{self.timestamp}.json"
        )
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for num_decks, result in results.items():
            serializable_results[str(num_decks)] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in result.items()
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _plot_comparison_results(self, results: Dict):
        """Create visualization plots for comparison results"""
        plt.style.use('default')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Blackjack AI Performance Comparison', fontsize=16)
        
        colors = ['#2196F3', '#4CAF50', '#FFC107', '#E91E63']
        
        # 1. Win Rates Comparison
        decks = []
        rates = []
        for num_decks, result in results.items():
            decks.append(num_decks)
            rates.append(result['evaluation_stats']['win_rate'])
        
        axes[0, 0].plot(decks, rates, 'o-', linewidth=2, color=colors[0])
        axes[0, 0].set_xlabel('Number of Decks')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_title('Win Rate vs Number of Decks')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROI Comparison
        rois = [result['evaluation_stats']['roi'] for result in results.values()]
        
        axes[0, 1].plot(decks, rois, 'o-', linewidth=2, color=colors[1])
        axes[0, 1].set_xlabel('Number of Decks')
        axes[0, 1].set_ylabel('ROI (%)')
        axes[0, 1].set_title('Return on Investment vs Number of Decks')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Decision Distribution
        hit_ratios = []
        stand_ratios = []
        for result in results.values():
            decisions = result['evaluation_stats']['decisions']
            hit_ratios.append(decisions['hit_ratio'])
            stand_ratios.append(decisions['stand_ratio'])
        
        axes[1, 0].plot(decks, hit_ratios, 'o-', label='Hit', color=colors[0])
        axes[1, 0].plot(decks, stand_ratios, 'o-', label='Stand', color=colors[1])
        axes[1, 0].set_xlabel('Number of Decks')
        axes[1, 0].set_ylabel('Decision Ratio')
        axes[1, 0].set_title('Decision Distribution vs Number of Decks')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. AI vs Baseline Comparison
        ai_rates = [result['evaluation_stats']['win_rate'] for result in results.values()]
        baseline_rates = [result['baseline_stats']['win_rate'] for result in results.values()]
        
        axes[1, 1].plot(decks, ai_rates, 'o-', label='AI Strategy', 
                       linewidth=2, color=colors[0])
        axes[1, 1].plot(decks, baseline_rates, 'o--', label='Baseline Strategy', 
                       linewidth=2, color=colors[1])
        axes[1, 1].set_xlabel('Number of Decks')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_title('AI vs Baseline Strategy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.save_dir,
            f'performance_comparison_{self.timestamp}.png'
        ))
        plt.close()

def run_simulation():
    """Run a complete simulation with default parameters"""
    simulator = BlackjackSimulator()
    
    # Test with common deck sizes used in casinos
    deck_sizes = [1, 2, 4, 6, 8]
    results = simulator.run_deck_comparison(
        deck_sizes=deck_sizes,
        episodes_per_size=10000,
        parallel=True
    )
    
    # Print summary
    print("\nSimulation Summary:")
    print("-" * 50)
    for num_decks, result in results.items():
        eval_stats = result['evaluation_stats']
        baseline_stats = result['baseline_stats']
        print(f"\n{num_decks} Deck(s):")
        print(f"AI Win Rate: {eval_stats['win_rate']:.2%}")
        print(f"Baseline Win Rate: {baseline_stats['win_rate']:.2%}")
        print(f"ROI: {eval_stats['roi']:.2f}%")
        print(f"Hit Ratio: {eval_stats['decisions']['hit_ratio']:.2%}")
        print(f"Stand Ratio: {eval_stats['decisions']['stand_ratio']:.2%}")

if __name__ == "__main__":
    run_simulation()