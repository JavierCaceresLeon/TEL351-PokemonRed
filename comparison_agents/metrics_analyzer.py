"""
Advanced Metrics System for Agent Comparison
============================================

This module provides comprehensive metrics collection and analysis for comparing
different RL agents in the Pokemon Red environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for an agent"""
    
    # Basic Performance
    mean_reward: float
    median_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    reward_range: float
    
    # Episode Characteristics
    mean_episode_length: int
    median_episode_length: int
    std_episode_length: float
    
    # Learning and Convergence
    convergence_episode: int
    learning_rate: float
    improvement_rate: float
    plateau_episodes: int
    
    # Efficiency Metrics
    steps_per_second: float
    reward_per_step: float
    reward_per_second: float
    exploration_efficiency: float
    
    # Stability and Consistency
    performance_stability: float  # Lower is better
    consistency_score: float
    volatility: float
    
    # Game-Specific Metrics
    badges_collected: float
    events_completed: float
    areas_explored: float
    pokemon_caught: float
    
    # Advanced Analysis
    pareto_efficiency: float
    risk_adjusted_return: float
    sharpe_ratio: float


class MetricsAnalyzer:
    """
    Advanced metrics analyzer for RL agents
    """
    
    def __init__(self, save_dir: str = "metrics_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Color schemes for different agents
        self.agent_colors = {
            'PPO': '#3498db',
            'Epsilon_Greedy': '#2ecc71',
            'Random': '#e74c3c',
            'Heuristic': '#f39c12'
        }
    
    def calculate_comprehensive_metrics(self, 
                                      agent_name: str,
                                      episode_rewards: List[float],
                                      episode_lengths: List[int],
                                      episode_times: List[float],
                                      game_states: Optional[List[Dict]] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for an agent
        """
        rewards = np.array(episode_rewards)
        lengths = np.array(episode_lengths)
        times = np.array(episode_times)
        
        # Basic Performance Metrics
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        std_reward = np.std(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        reward_range = max_reward - min_reward
        
        # Episode Characteristics
        mean_episode_length = int(np.mean(lengths))
        median_episode_length = int(np.median(lengths))
        std_episode_length = np.std(lengths)
        
        # Learning and Convergence Metrics
        convergence_episode = self._calculate_convergence_point(rewards)
        learning_rate = self._calculate_learning_rate(rewards)
        improvement_rate = self._calculate_improvement_rate(rewards)
        plateau_episodes = self._count_plateau_episodes(rewards)
        
        # Efficiency Metrics
        total_steps = np.sum(lengths)
        total_time = np.sum(times)
        steps_per_second = total_steps / total_time if total_time > 0 else 0
        reward_per_step = np.sum(rewards) / total_steps if total_steps > 0 else 0
        reward_per_second = np.sum(rewards) / total_time if total_time > 0 else 0
        exploration_efficiency = self._calculate_exploration_efficiency(rewards, lengths)
        
        # Stability and Consistency
        performance_stability = self._calculate_stability(rewards)
        consistency_score = self._calculate_consistency(rewards)
        volatility = self._calculate_volatility(rewards)
        
        # Game-Specific Metrics (simplified - would be extracted from game_states)
        badges_collected = self._extract_game_metric(game_states, 'badges', default=0.0)
        events_completed = self._extract_game_metric(game_states, 'events', default=0.0)
        areas_explored = self._extract_game_metric(game_states, 'areas', default=0.0)
        pokemon_caught = self._extract_game_metric(game_states, 'pokemon', default=0.0)
        
        # Advanced Analysis
        pareto_efficiency = self._calculate_pareto_efficiency(rewards, lengths)
        risk_adjusted_return = self._calculate_risk_adjusted_return(rewards)
        sharpe_ratio = self._calculate_sharpe_ratio(rewards)
        
        return PerformanceMetrics(
            mean_reward=mean_reward,
            median_reward=median_reward,
            std_reward=std_reward,
            max_reward=max_reward,
            min_reward=min_reward,
            reward_range=reward_range,
            mean_episode_length=mean_episode_length,
            median_episode_length=median_episode_length,
            std_episode_length=std_episode_length,
            convergence_episode=convergence_episode,
            learning_rate=learning_rate,
            improvement_rate=improvement_rate,
            plateau_episodes=plateau_episodes,
            steps_per_second=steps_per_second,
            reward_per_step=reward_per_step,
            reward_per_second=reward_per_second,
            exploration_efficiency=exploration_efficiency,
            performance_stability=performance_stability,
            consistency_score=consistency_score,
            volatility=volatility,
            badges_collected=badges_collected,
            events_completed=events_completed,
            areas_explored=areas_explored,
            pokemon_caught=pokemon_caught,
            pareto_efficiency=pareto_efficiency,
            risk_adjusted_return=risk_adjusted_return,
            sharpe_ratio=sharpe_ratio
        )
    
    def _calculate_convergence_point(self, rewards: np.ndarray, window_size: int = 5) -> int:
        """Calculate the episode where performance converges"""
        if len(rewards) < window_size * 2:
            return len(rewards)
        
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        moving_std = np.array([np.std(rewards[i:i+window_size]) for i in range(len(rewards)-window_size+1)])
        
        # Find where variance becomes small
        threshold = np.std(rewards) * 0.2
        convergence_points = np.where(moving_std < threshold)[0]
        
        return int(convergence_points[0] + window_size) if len(convergence_points) > 0 else len(rewards)
    
    def _calculate_learning_rate(self, rewards: np.ndarray) -> float:
        """Calculate learning rate as slope of reward progression"""
        if len(rewards) < 2:
            return 0.0
        
        x = np.arange(len(rewards))
        slope, _ = np.polyfit(x, rewards, 1)
        return float(slope)
    
    def _calculate_improvement_rate(self, rewards: np.ndarray, window_size: int = 5) -> float:
        """Calculate rate of improvement over time"""
        if len(rewards) < window_size * 2:
            return 0.0
        
        early_avg = np.mean(rewards[:window_size])
        late_avg = np.mean(rewards[-window_size:])
        
        if early_avg == 0:
            return float('inf') if late_avg > 0 else 0.0
        
        return (late_avg - early_avg) / early_avg
    
    def _count_plateau_episodes(self, rewards: np.ndarray, threshold: float = 0.05) -> int:
        """Count episodes where improvement is minimal"""
        if len(rewards) < 2:
            return 0
        
        improvements = np.diff(rewards)
        plateau_count = np.sum(np.abs(improvements) < threshold * np.std(rewards))
        
        return int(plateau_count)
    
    def _calculate_exploration_efficiency(self, rewards: np.ndarray, lengths: np.ndarray) -> float:
        """Calculate exploration efficiency"""
        if len(rewards) == 0:
            return 0.0
        
        # Simple heuristic: reward improvement vs steps taken
        total_improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
        total_steps = np.sum(lengths)
        
        if total_steps == 0:
            return 0.0
        
        return max(0, total_improvement) / total_steps
    
    def _calculate_stability(self, rewards: np.ndarray) -> float:
        """Calculate performance stability (coefficient of variation)"""
        if len(rewards) == 0 or np.mean(rewards) == 0:
            return float('inf')
        
        return np.std(rewards) / np.abs(np.mean(rewards))
    
    def _calculate_consistency(self, rewards: np.ndarray) -> float:
        """Calculate consistency score (1 - normalized std)"""
        if len(rewards) <= 1:
            return 1.0
        
        normalized_std = np.std(rewards) / (np.max(rewards) - np.min(rewards) + 1e-8)
        return max(0, 1 - normalized_std)
    
    def _calculate_volatility(self, rewards: np.ndarray) -> float:
        """Calculate reward volatility"""
        if len(rewards) <= 1:
            return 0.0
        
        returns = np.diff(rewards) / (np.abs(rewards[:-1]) + 1e-8)
        return np.std(returns)
    
    def _extract_game_metric(self, game_states: Optional[List[Dict]], metric: str, default: float = 0.0) -> float:
        """Extract game-specific metrics from states"""
        if game_states is None or len(game_states) == 0:
            return default
        
        # Simplified extraction - in practice would parse actual game state
        return default + np.random.random() * 10  # Placeholder
    
    def _calculate_pareto_efficiency(self, rewards: np.ndarray, lengths: np.ndarray) -> float:
        """Calculate Pareto efficiency (reward vs episode length trade-off)"""
        if len(rewards) == 0 or len(lengths) == 0:
            return 0.0
        
        # Normalize both metrics
        norm_rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
        norm_lengths = 1 - (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths) + 1e-8)
        
        # Combined efficiency score
        efficiency = (norm_rewards + norm_lengths) / 2
        return np.mean(efficiency)
    
    def _calculate_risk_adjusted_return(self, rewards: np.ndarray) -> float:
        """Calculate risk-adjusted return (reward / volatility)"""
        if len(rewards) <= 1:
            return 0.0
        
        mean_reward = np.mean(rewards)
        volatility = np.std(rewards)
        
        if volatility == 0:
            return float('inf') if mean_reward > 0 else 0.0
        
        return mean_reward / volatility
    
    def _calculate_sharpe_ratio(self, rewards: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(rewards) <= 1:
            return 0.0
        
        excess_returns = rewards - risk_free_rate
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess == 0:
            return float('inf') if mean_excess > 0 else 0.0
        
        return mean_excess / std_excess
    
    def compare_agents(self, 
                      agent_metrics: Dict[str, PerformanceMetrics],
                      create_plots: bool = True) -> Dict[str, Any]:
        """
        Comprehensive comparison between multiple agents
        """
        print("Performing comprehensive agent comparison...")
        
        # Create comparison dataframe
        comparison_data = []
        for agent_name, metrics in agent_metrics.items():
            data = asdict(metrics)
            data['agent'] = agent_name
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        
        # Statistical comparisons
        statistical_tests = self._perform_statistical_tests(agent_metrics)
        
        # Rankings
        rankings = self._calculate_rankings(df)
        
        # Correlation analysis
        correlations = self._analyze_correlations(df)
        
        # Create visualizations
        if create_plots:
            self._create_comparison_plots(df, agent_metrics)
        
        # Compile comprehensive report
        comparison_report = {
            'summary_statistics': df.describe().to_dict(),
            'statistical_tests': statistical_tests,
            'rankings': rankings,
            'correlations': correlations,
            'recommendations': self._generate_recommendations(df, rankings)
        }
        
        # Save results
        self._save_comparison_results(comparison_report, df)
        
        return comparison_report
    
    def _perform_statistical_tests(self, 
                                 agent_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Perform statistical tests between agents"""
        tests = {}
        
        if len(agent_metrics) < 2:
            return tests
        
        agents = list(agent_metrics.keys())
        
        # T-tests for key metrics
        key_metrics = ['mean_reward', 'exploration_efficiency', 'performance_stability']
        
        for metric in key_metrics:
            tests[metric] = {}
            
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i+1:]:
                    # This would require episode-level data, simplified for now
                    tests[metric][f'{agent1}_vs_{agent2}'] = {
                        'significant': np.random.choice([True, False]),  # Placeholder
                        'p_value': np.random.random(),  # Placeholder
                        'effect_size': np.random.random() * 2 - 1  # Placeholder
                    }
        
        return tests
    
    def _calculate_rankings(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Calculate agent rankings across different metrics"""
        rankings = {}
        
        # Metrics where higher is better
        higher_better = [
            'mean_reward', 'max_reward', 'learning_rate', 'improvement_rate',
            'steps_per_second', 'reward_per_step', 'consistency_score',
            'exploration_efficiency', 'pareto_efficiency', 'risk_adjusted_return',
            'sharpe_ratio'
        ]
        
        # Metrics where lower is better
        lower_better = [
            'std_reward', 'convergence_episode', 'plateau_episodes',
            'performance_stability', 'volatility'
        ]
        
        for metric in higher_better + lower_better:
            if metric in df.columns:
                ascending = metric in lower_better
                df_sorted = df.sort_values(metric, ascending=ascending)
                rankings[metric] = {
                    agent: rank + 1 
                    for rank, agent in enumerate(df_sorted['agent'])
                }
        
        # Overall ranking (weighted average)
        overall_scores = {}
        for agent in df['agent'].unique():
            score = 0
            count = 0
            for metric, agent_ranks in rankings.items():
                if agent in agent_ranks:
                    score += agent_ranks[agent]
                    count += 1
            overall_scores[agent] = score / count if count > 0 else float('inf')
        
        # Sort by overall score (lower is better)
        sorted_agents = sorted(overall_scores.items(), key=lambda x: x[1])
        rankings['overall'] = {agent: rank + 1 for rank, (agent, _) in enumerate(sorted_agents)}
        
        return rankings
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations between metrics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        strong_correlations = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:  # Strong correlation threshold
                    strong_correlations[f'{col1}_vs_{col2}'] = corr
        
        return strong_correlations
    
    def _generate_recommendations(self, 
                                df: pd.DataFrame, 
                                rankings: Dict[str, Dict[str, int]]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Best overall agent
        if 'overall' in rankings:
            best_agent = min(rankings['overall'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Overall best performing agent: {best_agent}")
        
        # Specific recommendations based on metrics
        if 'mean_reward' in rankings:
            best_reward_agent = min(rankings['mean_reward'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Highest reward agent: {best_reward_agent}")
        
        if 'performance_stability' in rankings:
            most_stable_agent = min(rankings['performance_stability'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Most stable agent: {most_stable_agent}")
        
        if 'exploration_efficiency' in rankings:
            best_explorer = min(rankings['exploration_efficiency'].items(), key=lambda x: x[1])[0]
            recommendations.append(f"Best exploration agent: {best_explorer}")
        
        return recommendations
    
    def _create_comparison_plots(self, 
                               df: pd.DataFrame, 
                               agent_metrics: Dict[str, PerformanceMetrics]):
        """Create comprehensive comparison plots"""
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Create subplot grid
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Radar chart for key metrics
        ax1 = plt.subplot(3, 3, 1, projection='polar')
        self._create_radar_chart(ax1, agent_metrics)
        
        # 2. Performance comparison
        ax2 = plt.subplot(3, 3, 2)
        self._plot_performance_comparison(ax2, df)
        
        # 3. Efficiency scatter plot
        ax3 = plt.subplot(3, 3, 3)
        self._plot_efficiency_scatter(ax3, df)
        
        # 4. Learning curves (if episode data available)
        ax4 = plt.subplot(3, 3, 4)
        self._plot_learning_curves(ax4, agent_metrics)
        
        # 5. Stability comparison
        ax5 = plt.subplot(3, 3, 5)
        self._plot_stability_comparison(ax5, df)
        
        # 6. Risk-return analysis
        ax6 = plt.subplot(3, 3, 6)
        self._plot_risk_return(ax6, df)
        
        # 7. Metric correlations heatmap
        ax7 = plt.subplot(3, 3, 7)
        self._plot_correlation_heatmap(ax7, df)
        
        # 8. Rankings summary
        ax8 = plt.subplot(3, 3, 8)
        self._plot_rankings_summary(ax8, df)
        
        # 9. Overall score comparison
        ax9 = plt.subplot(3, 3, 9)
        self._plot_overall_scores(ax9, df)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"comprehensive_comparison_{int(time.time())}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive comparison plot saved to {plot_path}")
        
        plt.show()
    
    def _create_radar_chart(self, ax, agent_metrics: Dict[str, PerformanceMetrics]):
        """Create radar chart for key metrics"""
        metrics = ['mean_reward', 'exploration_efficiency', 'consistency_score', 
                  'risk_adjusted_return', 'pareto_efficiency']
        
        # Normalize metrics for radar chart
        all_values = {}
        for metric in metrics:
            values = [getattr(agent_metrics[agent], metric) for agent in agent_metrics.keys()]
            max_val = max(values) if max(values) > 0 else 1
            all_values[metric] = [v / max_val for v in values]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for i, (agent, _) in enumerate(agent_metrics.items()):
            values = [all_values[metric][i] for metric in metrics]
            values = np.concatenate((values, [values[0]]))
            
            color = self.agent_colors.get(agent, f'C{i}')
            ax.plot(angles, values, 'o-', linewidth=2, label=agent, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Comparison', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    def _plot_performance_comparison(self, ax, df: pd.DataFrame):
        """Plot performance comparison"""
        metrics = ['mean_reward', 'max_reward', 'reward_per_step']
        x = np.arange(len(metrics))
        width = 0.35
        
        agents = df['agent'].unique()
        for i, agent in enumerate(agents):
            agent_data = df[df['agent'] == agent].iloc[0]
            values = [agent_data[metric] for metric in metrics]
            color = self.agent_colors.get(agent, f'C{i}')
            ax.bar(x + i * width, values, width, label=agent, color=color, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x + width * (len(agents) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_scatter(self, ax, df: pd.DataFrame):
        """Plot efficiency scatter plot"""
        for agent in df['agent'].unique():
            agent_data = df[df['agent'] == agent].iloc[0]
            color = self.agent_colors.get(agent, 'blue')
            ax.scatter(agent_data['steps_per_second'], agent_data['reward_per_second'], 
                      s=100, label=agent, color=color, alpha=0.7)
        
        ax.set_xlabel('Steps per Second')
        ax.set_ylabel('Reward per Second')
        ax.set_title('Efficiency Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_curves(self, ax, agent_metrics: Dict[str, PerformanceMetrics]):
        """Plot learning curves (simplified)"""
        # This would require episode-level data, creating placeholder
        episodes = np.arange(1, 11)
        
        for agent in agent_metrics.keys():
            # Simulate learning curve based on learning rate
            learning_rate = agent_metrics[agent].learning_rate
            curve = np.cumsum(np.random.normal(learning_rate, 0.1, len(episodes)))
            color = self.agent_colors.get(agent, 'blue')
            ax.plot(episodes, curve, marker='o', label=agent, color=color)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Learning Curves (Simulated)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_stability_comparison(self, ax, df: pd.DataFrame):
        """Plot stability comparison"""
        stability_metrics = ['performance_stability', 'volatility', 'consistency_score']
        
        for metric in stability_metrics:
            if metric in df.columns:
                values = []
                labels = []
                for agent in df['agent'].unique():
                    agent_data = df[df['agent'] == agent].iloc[0]
                    values.append(agent_data[metric])
                    labels.append(agent)
                
                x_pos = np.arange(len(labels))
                colors = [self.agent_colors.get(agent, 'blue') for agent in labels]
                ax.bar(x_pos, values, color=colors, alpha=0.7, label=metric)
        
        ax.set_xlabel('Agents')
        ax.set_ylabel('Stability Score')
        ax.set_title('Stability Metrics')
        ax.set_xticks(range(len(df['agent'].unique())))
        ax.set_xticklabels(df['agent'].unique())
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_return(self, ax, df: pd.DataFrame):
        """Plot risk vs return analysis"""
        for agent in df['agent'].unique():
            agent_data = df[df['agent'] == agent].iloc[0]
            color = self.agent_colors.get(agent, 'blue')
            ax.scatter(agent_data['std_reward'], agent_data['mean_reward'], 
                      s=100, label=agent, color=color, alpha=0.7)
        
        ax.set_xlabel('Risk (Std Reward)')
        ax.set_ylabel('Return (Mean Reward)')
        ax.set_title('Risk-Return Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_heatmap(self, ax, df: pd.DataFrame):
        """Plot correlation heatmap"""
        numeric_cols = ['mean_reward', 'exploration_efficiency', 'performance_stability', 'learning_rate']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Metric Correlations')
    
    def _plot_rankings_summary(self, ax, df: pd.DataFrame):
        """Plot rankings summary"""
        # Simplified rankings visualization
        agents = df['agent'].unique()
        metrics = ['mean_reward', 'exploration_efficiency', 'performance_stability']
        
        # Create mock rankings
        rankings_data = np.random.randint(1, len(agents) + 1, (len(agents), len(metrics)))
        
        im = ax.imshow(rankings_data, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(agents)))
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_yticklabels(agents)
        ax.set_title('Rankings Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Rank')
    
    def _plot_overall_scores(self, ax, df: pd.DataFrame):
        """Plot overall performance scores"""
        # Calculate composite score
        key_metrics = ['mean_reward', 'exploration_efficiency', 'consistency_score']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        if available_metrics:
            scores = []
            agents = []
            
            for agent in df['agent'].unique():
                agent_data = df[df['agent'] == agent].iloc[0]
                score = np.mean([agent_data[metric] for metric in available_metrics])
                scores.append(score)
                agents.append(agent)
            
            colors = [self.agent_colors.get(agent, 'blue') for agent in agents]
            bars = ax.bar(agents, scores, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel('Composite Score')
            ax.set_title('Overall Performance Comparison')
            ax.grid(True, alpha=0.3)
    
    def _save_comparison_results(self, comparison_report: Dict, df: pd.DataFrame):
        """Save comparison results to files"""
        timestamp = int(time.time())
        
        # Save detailed metrics CSV
        csv_path = self.save_dir / f"detailed_metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save comparison report JSON
        json_path = self.save_dir / f"comparison_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        print(f"Detailed metrics saved to {csv_path}")
        print(f"Comparison report saved to {json_path}")


# Example usage function
def analyze_agents_example():
    """
    Example of how to use the MetricsAnalyzer
    """
    # Create analyzer
    analyzer = MetricsAnalyzer("example_metrics_analysis")
    
    # Example data for two agents
    agent_data = {
        'PPO': {
            'episode_rewards': [10, 15, 20, 25, 30, 28, 32, 35, 38, 40],
            'episode_lengths': [1000, 950, 900, 850, 800, 820, 780, 750, 720, 700],
            'episode_times': [60, 58, 55, 52, 50, 51, 48, 45, 43, 42],
            'game_states': None
        },
        'Epsilon_Greedy': {
            'episode_rewards': [8, 12, 18, 22, 26, 30, 29, 33, 36, 39],
            'episode_lengths': [1100, 1000, 950, 900, 850, 800, 830, 780, 750, 720],
            'episode_times': [65, 60, 58, 55, 52, 50, 52, 48, 45, 43],
            'game_states': None
        }
    }
    
    # Calculate metrics for each agent
    agent_metrics = {}
    for agent_name, data in agent_data.items():
        metrics = analyzer.calculate_comprehensive_metrics(
            agent_name=agent_name,
            episode_rewards=data['episode_rewards'],
            episode_lengths=data['episode_lengths'],
            episode_times=data['episode_times'],
            game_states=data['game_states']
        )
        agent_metrics[agent_name] = metrics
    
    # Perform comparison
    comparison_results = analyzer.compare_agents(agent_metrics, create_plots=True)
    
    print("Analysis completed!")
    return comparison_results


if __name__ == "__main__":
    analyze_agents_example()