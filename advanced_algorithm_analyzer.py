#!/usr/bin/env python3
"""
Advanced Professional Algorithm Comparison and Visualization System
=================================================================

This system provides comprehensive comparison, analysis, and visualization
of all implemented algorithms for Pokemon Red environment with professional
quality standards and advanced statistical analysis.

Features:
- Complete algorithm performance comparison
- Advanced statistical analysis and testing
- Professional-grade visualizations
- Detailed pros/cons analysis
- Performance profiling and recommendations
- Publication-ready reports and graphics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Professional visualization configuration
plt.style.use('default')
COLORS = plt.cm.Set1(np.linspace(0, 1, 12))
PLASMA_COLORS = plt.cm.plasma(np.linspace(0, 1, 12))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

@dataclass
class AlgorithmAnalysis:
    """Comprehensive analysis results for an algorithm"""
    name: str
    category: str
    performance_metrics: Dict[str, float]
    statistical_analysis: Dict[str, Any]
    strengths: List[str]
    weaknesses: List[str]
    optimal_scenarios: List[str]
    complexity_analysis: Dict[str, str]
    
@dataclass
class ComparisonResult:
    """Results from comprehensive algorithm comparison"""
    algorithm_analyses: List[AlgorithmAnalysis]
    comparative_statistics: Dict[str, Any]
    recommendations: Dict[str, str]
    visualizations_generated: List[str]

class AdvancedAlgorithmAnalyzer:
    """Advanced analyzer for comprehensive algorithm comparison"""
    
    def __init__(self, results_dir: str = "RESULTADOS"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("advanced_comparison_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Algorithm categories and properties
        self.algorithm_categories = {
            'Reinforcement Learning': ['ppo'],
            'Probabilistic Search': ['epsilon_greedy_alta_exploracion', 'epsilon_greedy_balanceada', 'epsilon_greedy_conservadora'],
            'Informed Search': ['astar'],
            'Uninformed Search': ['bfs'],
            'Metaheuristic': ['tabu_search', 'simulated_annealing'],
            'Local Search': ['hill_climbing_steepest', 'hill_climbing_first', 'hill_climbing_restart']
        }
        
        # Algorithm properties for analysis
        self.algorithm_properties = {
            'ppo': {
                'complexity': 'O(n²)',
                'memory': 'High',
                'convergence': 'Guaranteed (policy)',
                'exploration': 'Adaptive',
                'optimality': 'Local optimum'
            },
            'epsilon_greedy_alta_exploracion': {
                'complexity': 'O(1)',
                'memory': 'Low',
                'convergence': 'Probabilistic',
                'exploration': 'High',
                'optimality': 'Suboptimal'
            },
            'epsilon_greedy_balanceada': {
                'complexity': 'O(1)',
                'memory': 'Low',
                'convergence': 'Probabilistic',
                'exploration': 'Balanced',
                'optimality': 'Near-optimal'
            },
            'epsilon_greedy_conservadora': {
                'complexity': 'O(1)',
                'memory': 'Low',
                'convergence': 'Probabilistic',
                'exploration': 'Low',
                'optimality': 'Exploitation-focused'
            },
            'astar': {
                'complexity': 'O(b^d)',
                'memory': 'High',
                'convergence': 'Guaranteed',
                'exploration': 'Systematic',
                'optimality': 'Optimal'
            },
            'bfs': {
                'complexity': 'O(b^d)',
                'memory': 'Very High',
                'convergence': 'Guaranteed',
                'exploration': 'Exhaustive',
                'optimality': 'Optimal'
            },
            'tabu_search': {
                'complexity': 'O(n²)',
                'memory': 'Medium',
                'convergence': 'No guarantee',
                'exploration': 'Adaptive',
                'optimality': 'Near-optimal'
            },
            'simulated_annealing': {
                'complexity': 'O(n)',
                'memory': 'Low',
                'convergence': 'Probabilistic',
                'exploration': 'Temperature-based',
                'optimality': 'Near-optimal'
            },
            'hill_climbing_steepest': {
                'complexity': 'O(n)',
                'memory': 'Low',
                'convergence': 'Local optimum',
                'exploration': 'Greedy',
                'optimality': 'Local optimum'
            },
            'hill_climbing_first': {
                'complexity': 'O(n)',
                'memory': 'Low',
                'convergence': 'Local optimum',
                'exploration': 'First improvement',
                'optimality': 'Local optimum'
            },
            'hill_climbing_restart': {
                'complexity': 'O(n*k)',
                'memory': 'Low',
                'convergence': 'Better local optimum',
                'exploration': 'Multiple restarts',
                'optimality': 'Better local optimum'
            }
        }

    def load_all_algorithm_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from all algorithm executions"""
        algorithm_data = {}
        
        # Load from enhanced execution results
        enhanced_dir = self.results_dir / "enhanced_execution"
        if enhanced_dir.exists():
            for algo_dir in enhanced_dir.iterdir():
                if algo_dir.is_dir():
                    algo_name = algo_dir.name
                    summary_files = list(algo_dir.glob("*summary*.csv"))
                    
                    if summary_files:
                        try:
                            df = pd.read_csv(summary_files[0])
                            algorithm_data[algo_name] = df
                            print(f"Loaded data for {algo_name}: {len(df)} records")
                        except Exception as e:
                            print(f"Error loading {algo_name}: {e}")
        
        return algorithm_data

    def analyze_algorithm_performance(self, data: Dict[str, pd.DataFrame]) -> List[AlgorithmAnalysis]:
        """Perform comprehensive analysis of each algorithm"""
        analyses = []
        
        for algo_name, df in data.items():
            if df.empty:
                continue
                
            # Extract performance metrics
            metrics = self._extract_performance_metrics(df)
            
            # Statistical analysis
            stats_analysis = self._perform_statistical_analysis(df)
            
            # Get algorithm category
            category = self._get_algorithm_category(algo_name)
            
            # Define strengths and weaknesses
            strengths, weaknesses = self._analyze_strengths_weaknesses(algo_name, metrics)
            
            # Optimal scenarios
            optimal_scenarios = self._define_optimal_scenarios(algo_name)
            
            # Complexity analysis
            complexity = self.algorithm_properties.get(algo_name, {})
            
            analysis = AlgorithmAnalysis(
                name=algo_name,
                category=category,
                performance_metrics=metrics,
                statistical_analysis=stats_analysis,
                strengths=strengths,
                weaknesses=weaknesses,
                optimal_scenarios=optimal_scenarios,
                complexity_analysis=complexity
            )
            
            analyses.append(analysis)
            
        return analyses

    def _extract_performance_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract key performance metrics from algorithm data"""
        metrics = {}
        
        # Standard metrics
        if 'Pasos Totales' in df.columns:
            metrics['avg_steps'] = df['Pasos Totales'].mean()
            metrics['std_steps'] = df['Pasos Totales'].std()
            metrics['min_steps'] = df['Pasos Totales'].min()
            metrics['max_steps'] = df['Pasos Totales'].max()
        
        if 'Tiempo (s)' in df.columns:
            metrics['avg_time'] = df['Tiempo (s)'].mean()
            metrics['std_time'] = df['Tiempo (s)'].std()
            
        if 'Recompensa Total' in df.columns:
            metrics['avg_reward'] = df['Recompensa Total'].mean()
            metrics['std_reward'] = df['Recompensa Total'].std()
            
        # Efficiency metrics
        if 'avg_steps' in metrics and 'avg_time' in metrics and metrics['avg_time'] > 0:
            metrics['efficiency'] = metrics['avg_steps'] / metrics['avg_time']
            
        if 'avg_reward' in metrics and 'avg_steps' in metrics and metrics['avg_steps'] > 0:
            metrics['reward_efficiency'] = metrics['avg_reward'] / metrics['avg_steps']
        
        return metrics

    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on algorithm performance"""
        analysis = {}
        
        if 'Pasos Totales' in df.columns:
            steps = df['Pasos Totales'].dropna()
            if len(steps) > 1:
                analysis['steps_normality'] = stats.shapiro(steps)[1] > 0.05
                analysis['steps_variance'] = steps.var()
                analysis['steps_cv'] = steps.std() / steps.mean() if steps.mean() > 0 else 0
                
        if 'Recompensa Total' in df.columns:
            rewards = df['Recompensa Total'].dropna()
            if len(rewards) > 1:
                analysis['reward_normality'] = stats.shapiro(rewards)[1] > 0.05
                analysis['reward_stability'] = 1 - (rewards.std() / rewards.mean()) if rewards.mean() > 0 else 0
                
        return analysis

    def _get_algorithm_category(self, algo_name: str) -> str:
        """Get the category of an algorithm"""
        for category, algorithms in self.algorithm_categories.items():
            if any(algo in algo_name for algo in algorithms):
                return category
        return "Unknown"

    def _analyze_strengths_weaknesses(self, algo_name: str, metrics: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Analyze algorithm strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Algorithm-specific analysis
        if 'ppo' in algo_name:
            strengths = [
                "Adaptive learning from experience",
                "Handles complex state spaces",
                "Policy gradient optimization",
                "Good exploration-exploitation balance"
            ]
            weaknesses = [
                "High computational requirements",
                "Requires extensive training",
                "Sensitive to hyperparameters",
                "May converge to local optima"
            ]
        elif 'epsilon_greedy' in algo_name:
            strengths = [
                "Simple and efficient implementation",
                "Low computational overhead",
                "Probabilistic exploration",
                "Easy to tune and understand"
            ]
            weaknesses = [
                "No learning from experience",
                "Random exploration can be inefficient",
                "No memory of past actions",
                "Performance depends on epsilon tuning"
            ]
            
            if 'alta_exploracion' in algo_name:
                strengths.append("High exploration of new areas")
                weaknesses.append("May be too random for exploitation")
            elif 'conservadora' in algo_name:
                strengths.append("Focused on exploitation")
                weaknesses.append("Limited exploration capabilities")
                
        elif 'astar' in algo_name:
            strengths = [
                "Optimal pathfinding guarantee",
                "Intelligent heuristic guidance",
                "Complete search algorithm",
                "Efficient for goal-directed tasks"
            ]
            weaknesses = [
                "High memory requirements",
                "Computationally expensive",
                "Requires good heuristic function",
                "May be slow in large spaces"
            ]
        elif 'bfs' in algo_name:
            strengths = [
                "Guarantees optimal solution",
                "Complete search algorithm",
                "Systematic exploration",
                "Simple implementation"
            ]
            weaknesses = [
                "Exponential time complexity",
                "Very high memory usage",
                "No heuristic guidance",
                "Inefficient for large spaces"
            ]
        elif 'tabu_search' in algo_name:
            strengths = [
                "Avoids local optima",
                "Memory-based search",
                "Adaptive neighborhood exploration",
                "Good for complex landscapes"
            ]
            weaknesses = [
                "No optimality guarantee",
                "Parameter tuning required",
                "Memory management complexity",
                "May cycle without proper controls"
            ]
        elif 'simulated_annealing' in algo_name:
            strengths = [
                "Probabilistic escape from local optima",
                "Temperature-controlled exploration",
                "Simple implementation",
                "Good for optimization problems"
            ]
            weaknesses = [
                "Cooling schedule sensitivity",
                "No optimality guarantee",
                "May accept poor solutions early",
                "Parameter tuning critical"
            ]
        elif 'hill_climbing' in algo_name:
            strengths = [
                "Simple and fast",
                "Low memory requirements",
                "Intuitive greedy approach"
            ]
            weaknesses = [
                "Gets stuck in local optima",
                "No exploration mechanism",
                "Sensitive to initial state"
            ]
            
            if 'restart' in algo_name:
                strengths.append("Multiple attempts improve results")
                
        return strengths, weaknesses

    def _define_optimal_scenarios(self, algo_name: str) -> List[str]:
        """Define optimal scenarios for each algorithm"""
        if 'ppo' in algo_name:
            return [
                "Complex environments with large state spaces",
                "Long-term planning requirements",
                "Environments with sparse rewards",
                "Continuous learning scenarios"
            ]
        elif 'epsilon_greedy' in algo_name:
            scenarios = [
                "Simple exploration tasks",
                "Real-time decision making",
                "Limited computational resources",
                "Baseline comparison scenarios"
            ]
            if 'alta_exploracion' in algo_name:
                scenarios.append("Unknown environments requiring exploration")
            elif 'conservadora' in algo_name:
                scenarios.append("Well-known environments for exploitation")
            return scenarios
        elif 'astar' in algo_name:
            return [
                "Goal-directed navigation",
                "Pathfinding in known environments",
                "Optimal solution requirements",
                "Grid-based or graph problems"
            ]
        elif 'bfs' in algo_name:
            return [
                "Small search spaces",
                "Guaranteed optimal solution needed",
                "Systematic exploration requirements",
                "Complete state enumeration"
            ]
        elif 'tabu_search' in algo_name:
            return [
                "Complex optimization landscapes",
                "Avoiding local optima critical",
                "Medium-sized search spaces",
                "Combinatorial optimization"
            ]
        elif 'simulated_annealing' in algo_name:
            return [
                "Global optimization problems",
                "Acceptable suboptimal solutions",
                "Continuous or discrete spaces",
                "Time-constrained optimization"
            ]
        elif 'hill_climbing' in algo_name:
            return [
                "Simple optimization tasks",
                "Quick local improvements",
                "Resource-constrained environments",
                "Local search problems"
            ]
        return ["General purpose scenarios"]

    def generate_comprehensive_visualizations(self, analyses: List[AlgorithmAnalysis]) -> List[str]:
        """Generate comprehensive professional visualizations"""
        visualizations = []
        
        # 1. Performance Overview Dashboard
        self._create_performance_dashboard(analyses)
        visualizations.append("performance_dashboard.png")
        
        # 2. Algorithm Category Analysis
        self._create_category_analysis(analyses)
        visualizations.append("category_analysis.png")
        
        # 3. Statistical Comparison
        self._create_statistical_comparison(analyses)
        visualizations.append("statistical_comparison.png")
        
        # 4. Complexity vs Performance Analysis
        self._create_complexity_analysis(analyses)
        visualizations.append("complexity_analysis.png")
        
        # 5. Strengths and Weaknesses Matrix
        self._create_strengths_weaknesses_matrix(analyses)
        visualizations.append("strengths_weaknesses_matrix.png")
        
        # 6. Recommendation Matrix
        self._create_recommendation_matrix(analyses)
        visualizations.append("recommendation_matrix.png")
        
        return visualizations

    def _create_performance_dashboard(self, analyses: List[AlgorithmAnalysis]):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data for plotting
        algo_names = [a.name for a in analyses]
        steps_data = [a.performance_metrics.get('avg_steps', 0) for a in analyses]
        time_data = [a.performance_metrics.get('avg_time', 0) for a in analyses]
        reward_data = [a.performance_metrics.get('avg_reward', 0) for a in analyses]
        efficiency_data = [a.performance_metrics.get('efficiency', 0) for a in analyses]
        
        # 1. Average Steps Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(range(len(algo_names)), steps_data, color=PLASMA_COLORS[:len(algo_names)])
        ax1.set_title('Average Steps to Completion', fontweight='bold')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Steps')
        ax1.set_xticks(range(len(algo_names)))
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in algo_names], rotation=45, ha='right')
        
        # 2. Execution Time Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(range(len(algo_names)), time_data, color=PLASMA_COLORS[:len(algo_names)])
        ax2.set_title('Average Execution Time', fontweight='bold')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xticks(range(len(algo_names)))
        ax2.set_xticklabels([name.replace('_', ' ').title() for name in algo_names], rotation=45, ha='right')
        
        # 3. Reward Performance
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(range(len(algo_names)), reward_data, color=PLASMA_COLORS[:len(algo_names)])
        ax3.set_title('Average Total Reward', fontweight='bold')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Reward')
        ax3.set_xticks(range(len(algo_names)))
        ax3.set_xticklabels([name.replace('_', ' ').title() for name in algo_names], rotation=45, ha='right')
        
        # 4. Efficiency Comparison
        ax4 = fig.add_subplot(gs[0, 3])
        bars4 = ax4.bar(range(len(algo_names)), efficiency_data, color=PLASMA_COLORS[:len(algo_names)])
        ax4.set_title('Efficiency (Steps/Second)', fontweight='bold')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Efficiency')
        ax4.set_xticks(range(len(algo_names)))
        ax4.set_xticklabels([name.replace('_', ' ').title() for name in algo_names], rotation=45, ha='right')
        
        # 5. Performance Radar Chart
        ax5 = fig.add_subplot(gs[1, :2], projection='polar')
        categories = ['Steps', 'Time', 'Reward', 'Efficiency']
        
        # Initialize angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Normalize data for radar chart
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        if len(analyses) > 0:
            radar_data = np.array([steps_data, time_data, reward_data, efficiency_data]).T
            # Handle cases where all values might be zero
            if radar_data.max() > 0:
                radar_data_norm = scaler.fit_transform(radar_data)
                
                for i, (analysis, color) in enumerate(zip(analyses[:5], PLASMA_COLORS[:5])):  # Limit to 5 for readability
                    values = np.concatenate((radar_data_norm[i], [radar_data_norm[i][0]]))
                    ax5.plot(angles, values, 'o-', linewidth=2, label=analysis.name.replace('_', ' ').title(), color=color)
                    ax5.fill(angles, values, alpha=0.1, color=color)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.set_title('Performance Radar Chart', fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 6. Category Performance
        ax6 = fig.add_subplot(gs[1, 2:])
        category_performance = {}
        for analysis in analyses:
            category = analysis.category
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(analysis.performance_metrics.get('avg_steps', 0))
        
        categories = list(category_performance.keys())
        category_means = [np.mean(category_performance[cat]) for cat in categories]
        
        bars6 = ax6.bar(categories, category_means, color=PLASMA_COLORS[:len(categories)])
        ax6.set_title('Performance by Algorithm Category', fontweight='bold')
        ax6.set_xlabel('Category')
        ax6.set_ylabel('Average Steps')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Detailed Performance Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        table_data = []
        for analysis in analyses:
            metrics = analysis.performance_metrics
            table_data.append([
                analysis.name.replace('_', ' ').title(),
                f"{metrics.get('avg_steps', 0):.1f}",
                f"{metrics.get('avg_time', 0):.2f}",
                f"{metrics.get('avg_reward', 0):.3f}",
                f"{metrics.get('efficiency', 0):.2f}",
                analysis.category
            ])
        
        table = ax7.table(
            cellText=table_data,
            colLabels=['Algorithm', 'Avg Steps', 'Avg Time (s)', 'Avg Reward', 'Efficiency', 'Category'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(6):
                cell = table[i, j]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.suptitle('Comprehensive Algorithm Performance Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(self.output_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_category_analysis(self, analyses: List[AlgorithmAnalysis]):
        """Create algorithm category analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group algorithms by category
        category_data = {}
        for analysis in analyses:
            category = analysis.category
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(analysis)
        
        # 1. Steps by Category (Box Plot)
        categories = list(category_data.keys())
        steps_by_category = []
        for category in categories:
            steps = [a.performance_metrics.get('avg_steps', 0) for a in category_data[category]]
            steps_by_category.append(steps)
        
        bp1 = ax1.boxplot(steps_by_category, labels=categories, patch_artist=True)
        for patch, color in zip(bp1['boxes'], PLASMA_COLORS):
            patch.set_facecolor(color)
        ax1.set_title('Steps Distribution by Algorithm Category', fontweight='bold')
        ax1.set_ylabel('Average Steps')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Time by Category
        time_by_category = []
        for category in categories:
            times = [a.performance_metrics.get('avg_time', 0) for a in category_data[category]]
            time_by_category.append(times)
        
        bp2 = ax2.boxplot(time_by_category, labels=categories, patch_artist=True)
        for patch, color in zip(bp2['boxes'], PLASMA_COLORS):
            patch.set_facecolor(color)
        ax2.set_title('Execution Time by Algorithm Category', fontweight='bold')
        ax2.set_ylabel('Average Time (s)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Category Size and Performance
        category_sizes = [len(category_data[cat]) for cat in categories]
        category_performance = [np.mean([a.performance_metrics.get('avg_steps', 0) for a in category_data[cat]]) for cat in categories]
        
        scatter = ax3.scatter(category_sizes, category_performance, 
                            c=range(len(categories)), cmap='plasma', s=200, alpha=0.7)
        for i, category in enumerate(categories):
            ax3.annotate(category, (category_sizes[i], category_performance[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax3.set_xlabel('Number of Algorithms in Category')
        ax3.set_ylabel('Average Performance (Steps)')
        ax3.set_title('Category Size vs Performance', fontweight='bold')
        
        # 4. Algorithm Complexity Distribution
        complexity_counts = {}
        for analysis in analyses:
            complexity = analysis.complexity_analysis.get('complexity', 'Unknown')
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        complexities = list(complexity_counts.keys())
        counts = list(complexity_counts.values())
        
        wedges, texts, autotexts = ax4.pie(counts, labels=complexities, autopct='%1.1f%%', 
                                          colors=PLASMA_COLORS[:len(complexities)])
        ax4.set_title('Algorithm Complexity Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_statistical_comparison(self, analyses: List[AlgorithmAnalysis]):
        """Create statistical comparison visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract statistical data
        algo_names = [a.name.replace('_', ' ').title() for a in analyses]
        
        # 1. Performance Variance Analysis
        variances = []
        cvs = []  # Coefficient of variation
        for analysis in analyses:
            stats_data = analysis.statistical_analysis
            variances.append(stats_data.get('steps_variance', 0))
            cvs.append(stats_data.get('steps_cv', 0))
        
        x_pos = np.arange(len(algo_names))
        bars1 = ax1.bar(x_pos, variances, color=PLASMA_COLORS[:len(algo_names)])
        ax1.set_title('Performance Variance by Algorithm', fontweight='bold')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Steps Variance')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(algo_names, rotation=45, ha='right')
        
        # 2. Coefficient of Variation (Stability)
        bars2 = ax2.bar(x_pos, cvs, color=PLASMA_COLORS[:len(algo_names)])
        ax2.set_title('Algorithm Stability (Lower CV = More Stable)', fontweight='bold')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(algo_names, rotation=45, ha='right')
        
        # 3. Performance vs Stability Scatter
        performance = [a.performance_metrics.get('avg_steps', 0) for a in analyses]
        scatter = ax3.scatter(performance, cvs, c=range(len(analyses)), 
                            cmap='plasma', s=100, alpha=0.7)
        for i, name in enumerate(algo_names):
            ax3.annotate(name[:10], (performance[i], cvs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Average Performance (Steps)')
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('Performance vs Stability Trade-off', fontweight='bold')
        
        # 4. Normality Test Results
        normality_results = []
        for analysis in analyses:
            is_normal = analysis.statistical_analysis.get('steps_normality', False)
            normality_results.append(1 if is_normal else 0)
        
        bars4 = ax4.bar(x_pos, normality_results, color=['green' if nr else 'red' for nr in normality_results])
        ax4.set_title('Performance Distribution Normality', fontweight='bold')
        ax4.set_xlabel('Algorithm')
        ax4.set_ylabel('Normal Distribution (1=Yes, 0=No)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(algo_names, rotation=45, ha='right')
        ax4.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_complexity_analysis(self, analyses: List[AlgorithmAnalysis]):
        """Create complexity vs performance analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract complexity and performance data
        complexity_levels = {
            'O(1)': 1, 'O(n)': 2, 'O(n²)': 3, 'O(n*k)': 3.5, 'O(b^d)': 4
        }
        
        complexities = []
        performances = []
        memory_usage = []
        convergence_types = []
        
        for analysis in analyses:
            complexity = analysis.complexity_analysis.get('complexity', 'O(n)')
            complexities.append(complexity_levels.get(complexity, 2))
            performances.append(analysis.performance_metrics.get('avg_steps', 0))
            
            memory = analysis.complexity_analysis.get('memory', 'Medium')
            memory_levels = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
            memory_usage.append(memory_levels.get(memory, 2))
            
            convergence_types.append(analysis.complexity_analysis.get('convergence', 'Unknown'))
        
        # 1. Complexity vs Performance
        scatter1 = ax1.scatter(complexities, performances, c=range(len(analyses)), 
                              cmap='plasma', s=100, alpha=0.7)
        for i, analysis in enumerate(analyses):
            ax1.annotate(analysis.name.replace('_', ' ')[:15], 
                        (complexities[i], performances[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Time Complexity Level')
        ax1.set_ylabel('Average Performance (Steps)')
        ax1.set_title('Time Complexity vs Performance', fontweight='bold')
        ax1.set_xticks(range(1, 5))
        ax1.set_xticklabels(['O(1)', 'O(n)', 'O(n²)', 'O(b^d)'])
        
        # 2. Memory Usage vs Performance
        scatter2 = ax2.scatter(memory_usage, performances, c=range(len(analyses)), 
                              cmap='plasma', s=100, alpha=0.7)
        for i, analysis in enumerate(analyses):
            ax2.annotate(analysis.name.replace('_', ' ')[:15], 
                        (memory_usage[i], performances[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Memory Usage Level')
        ax2.set_ylabel('Average Performance (Steps)')
        ax2.set_title('Memory Usage vs Performance', fontweight='bold')
        ax2.set_xticks(range(1, 5))
        ax2.set_xticklabels(['Low', 'Medium', 'High', 'Very High'])
        
        # 3. Convergence Guarantee Distribution
        convergence_counts = {}
        for conv_type in convergence_types:
            convergence_counts[conv_type] = convergence_counts.get(conv_type, 0) + 1
        
        conv_types = list(convergence_counts.keys())
        conv_counts = list(convergence_counts.values())
        
        bars3 = ax3.bar(conv_types, conv_counts, color=PLASMA_COLORS[:len(conv_types)])
        ax3.set_title('Convergence Guarantee Distribution', fontweight='bold')
        ax3.set_xlabel('Convergence Type')
        ax3.set_ylabel('Number of Algorithms')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Efficiency vs Complexity Trade-off
        efficiencies = [a.performance_metrics.get('efficiency', 0) for a in analyses]
        
        scatter4 = ax4.scatter(complexities, efficiencies, c=range(len(analyses)), 
                              cmap='plasma', s=100, alpha=0.7)
        for i, analysis in enumerate(analyses):
            ax4.annotate(analysis.name.replace('_', ' ')[:15], 
                        (complexities[i], efficiencies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Time Complexity Level')
        ax4.set_ylabel('Efficiency (Steps/Second)')
        ax4.set_title('Complexity vs Efficiency Trade-off', fontweight='bold')
        ax4.set_xticks(range(1, 5))
        ax4.set_xticklabels(['O(1)', 'O(n)', 'O(n²)', 'O(b^d)'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_strengths_weaknesses_matrix(self, analyses: List[AlgorithmAnalysis]):
        """Create strengths and weaknesses analysis matrix"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Prepare data for heatmap
        all_strengths = set()
        all_weaknesses = set()
        
        for analysis in analyses:
            all_strengths.update(analysis.strengths)
            all_weaknesses.update(analysis.weaknesses)
        
        # Limit to most common strengths/weaknesses for readability
        all_strengths = list(all_strengths)[:8]
        all_weaknesses = list(all_weaknesses)[:8]
        
        # Create strength matrix
        strength_matrix = np.zeros((len(analyses), len(all_strengths)))
        for i, analysis in enumerate(analyses):
            for j, strength in enumerate(all_strengths):
                if strength in analysis.strengths:
                    strength_matrix[i, j] = 1
        
        # Create weakness matrix
        weakness_matrix = np.zeros((len(analyses), len(all_weaknesses)))
        for i, analysis in enumerate(analyses):
            for j, weakness in enumerate(all_weaknesses):
                if weakness in analysis.weaknesses:
                    weakness_matrix[i, j] = 1
        
        # Plot strengths heatmap
        im1 = ax1.imshow(strength_matrix, cmap='Greens', aspect='auto')
        ax1.set_title('Algorithm Strengths Matrix', fontweight='bold', fontsize=14)
        ax1.set_xticks(range(len(all_strengths)))
        ax1.set_xticklabels([s[:20] + '...' if len(s) > 20 else s for s in all_strengths], 
                           rotation=45, ha='right', fontsize=9)
        ax1.set_yticks(range(len(analyses)))
        ax1.set_yticklabels([a.name.replace('_', ' ').title() for a in analyses], fontsize=10)
        
        # Add text annotations for strengths
        for i in range(len(analyses)):
            for j in range(len(all_strengths)):
                if strength_matrix[i, j] == 1:
                    ax1.text(j, i, '✓', ha='center', va='center', fontsize=16, fontweight='bold')
        
        # Plot weaknesses heatmap
        im2 = ax2.imshow(weakness_matrix, cmap='Reds', aspect='auto')
        ax2.set_title('Algorithm Weaknesses Matrix', fontweight='bold', fontsize=14)
        ax2.set_xticks(range(len(all_weaknesses)))
        ax2.set_xticklabels([w[:20] + '...' if len(w) > 20 else w for w in all_weaknesses], 
                           rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(len(analyses)))
        ax2.set_yticklabels([a.name.replace('_', ' ').title() for a in analyses], fontsize=10)
        
        # Add text annotations for weaknesses
        for i in range(len(analyses)):
            for j in range(len(all_weaknesses)):
                if weakness_matrix[i, j] == 1:
                    ax2.text(j, i, '✗', ha='center', va='center', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'strengths_weaknesses_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_recommendation_matrix(self, analyses: List[AlgorithmAnalysis]):
        """Create algorithm recommendation matrix for different scenarios"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Define scenarios
        scenarios = [
            "Real-time Decision Making",
            "Optimal Solution Required", 
            "Large State Spaces",
            "Limited Computational Resources",
            "Unknown Environments",
            "Goal-directed Navigation",
            "Complex Optimization",
            "Quick Local Improvements"
        ]
        
        # Create recommendation scores (0-5 scale)
        recommendation_matrix = np.zeros((len(analyses), len(scenarios)))
        
        for i, analysis in enumerate(analyses):
            algo_name = analysis.name
            
            # Score each algorithm for each scenario based on properties
            if 'ppo' in algo_name:
                scores = [2, 4, 5, 2, 5, 4, 5, 3]
            elif 'epsilon_greedy' in algo_name:
                base_scores = [5, 2, 3, 5, 4, 3, 2, 4]
                if 'alta_exploracion' in algo_name:
                    scores = [s + 1 if s_idx == 4 else s for s_idx, s in enumerate(base_scores)]  # Better for unknown environments
                elif 'conservadora' in algo_name:
                    scores = [s + 1 if s_idx == 0 else s for s_idx, s in enumerate(base_scores)]  # Better for real-time
                else:
                    scores = base_scores
            elif 'astar' in algo_name:
                scores = [3, 5, 3, 3, 3, 5, 4, 2]
            elif 'bfs' in algo_name:
                scores = [2, 5, 1, 1, 2, 4, 2, 1]
            elif 'tabu_search' in algo_name:
                scores = [3, 3, 4, 3, 4, 3, 5, 3]
            elif 'simulated_annealing' in algo_name:
                scores = [3, 4, 4, 4, 4, 3, 5, 3]
            elif 'hill_climbing' in algo_name:
                base_scores = [4, 2, 2, 5, 2, 2, 3, 5]
                if 'restart' in algo_name:
                    scores = [s + 1 if s_idx in [1, 6] else s for s_idx, s in enumerate(base_scores)]  # Better for optimization
                else:
                    scores = base_scores
            else:
                scores = [3] * len(scenarios)
            
            recommendation_matrix[i, :] = scores
        
        # Create heatmap
        im = ax.imshow(recommendation_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Recommendation Score (0-5)', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_title('Algorithm Recommendation Matrix by Scenario', fontweight='bold', fontsize=16, pad=20)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_yticks(range(len(analyses)))
        ax.set_yticklabels([a.name.replace('_', ' ').title() for a in analyses])
        
        # Add text annotations
        for i in range(len(analyses)):
            for j in range(len(scenarios)):
                score = recommendation_matrix[i, j]
                color = 'white' if score < 2.5 else 'black'
                ax.text(j, i, f'{score:.0f}', ha='center', va='center', 
                       fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recommendation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_comprehensive_report(self, analyses: List[AlgorithmAnalysis]) -> str:
        """Generate comprehensive analysis report"""
        report_path = self.output_dir / "comprehensive_algorithm_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Algorithm Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"This report analyzes {len(analyses)} algorithms across multiple dimensions including performance, "
                   f"complexity, strengths, weaknesses, and optimal use cases.\n\n")
            
            # Category overview
            categories = set(a.category for a in analyses)
            f.write(f"**Algorithm Categories Analyzed:** {', '.join(categories)}\n\n")
            
            # Performance summary
            best_performance = min(analyses, key=lambda a: a.performance_metrics.get('avg_steps', float('inf')))
            fastest_execution = min(analyses, key=lambda a: a.performance_metrics.get('avg_time', float('inf')))
            
            f.write(f"**Best Overall Performance:** {best_performance.name} "
                   f"({best_performance.performance_metrics.get('avg_steps', 0):.1f} avg steps)\n")
            f.write(f"**Fastest Execution:** {fastest_execution.name} "
                   f"({fastest_execution.performance_metrics.get('avg_time', 0):.2f} avg seconds)\n\n")
            
            # Detailed analysis for each algorithm
            f.write("## Detailed Algorithm Analysis\n\n")
            
            for analysis in analyses:
                f.write(f"### {analysis.name.replace('_', ' ').title()}\n\n")
                f.write(f"**Category:** {analysis.category}\n\n")
                
                # Performance metrics
                f.write("**Performance Metrics:**\n")
                metrics = analysis.performance_metrics
                f.write(f"- Average Steps: {metrics.get('avg_steps', 0):.1f}\n")
                f.write(f"- Average Time: {metrics.get('avg_time', 0):.2f}s\n")
                f.write(f"- Average Reward: {metrics.get('avg_reward', 0):.3f}\n")
                f.write(f"- Efficiency: {metrics.get('efficiency', 0):.2f} steps/second\n\n")
                
                # Complexity analysis
                f.write("**Complexity Analysis:**\n")
                complexity = analysis.complexity_analysis
                f.write(f"- Time Complexity: {complexity.get('complexity', 'N/A')}\n")
                f.write(f"- Memory Usage: {complexity.get('memory', 'N/A')}\n")
                f.write(f"- Convergence: {complexity.get('convergence', 'N/A')}\n")
                f.write(f"- Exploration Strategy: {complexity.get('exploration', 'N/A')}\n\n")
                
                # Strengths
                f.write("**Strengths:**\n")
                for strength in analysis.strengths:
                    f.write(f"- {strength}\n")
                f.write("\n")
                
                # Weaknesses
                f.write("**Weaknesses:**\n")
                for weakness in analysis.weaknesses:
                    f.write(f"- {weakness}\n")
                f.write("\n")
                
                # Optimal scenarios
                f.write("**Optimal Scenarios:**\n")
                for scenario in analysis.optimal_scenarios:
                    f.write(f"- {scenario}\n")
                f.write("\n")
                
                f.write("---\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### For Pokemon Red Environment:\n\n")
            
            # Best performers by category
            rl_algorithms = [a for a in analyses if a.category == 'Reinforcement Learning']
            if rl_algorithms:
                best_rl = min(rl_algorithms, key=lambda a: a.performance_metrics.get('avg_steps', float('inf')))
                f.write(f"**Best Reinforcement Learning Algorithm:** {best_rl.name}\n")
                f.write("- Recommended for: Long-term learning and adaptation\n\n")
            
            search_algorithms = [a for a in analyses if 'Search' in a.category]
            if search_algorithms:
                best_search = min(search_algorithms, key=lambda a: a.performance_metrics.get('avg_steps', float('inf')))
                f.write(f"**Best Search Algorithm:** {best_search.name}\n")
                f.write("- Recommended for: Goal-directed navigation tasks\n\n")
            
            # General recommendations
            f.write("### General Recommendations:\n\n")
            f.write("1. **For Real-time Applications:** Use epsilon-greedy variants for quick decisions\n")
            f.write("2. **For Optimal Solutions:** Use A* or BFS when solution quality is critical\n")
            f.write("3. **For Complex Environments:** Use PPO for learning and adaptation\n")
            f.write("4. **For Resource-Constrained Scenarios:** Use hill climbing or epsilon-greedy\n")
            f.write("5. **For Exploration Tasks:** Use high-exploration epsilon-greedy or tabu search\n\n")
            
            f.write("## Conclusion\n\n")
            f.write("The analysis reveals distinct trade-offs between different algorithmic approaches. "
                   "The choice of algorithm should be based on specific requirements including computational "
                   "constraints, solution quality needs, and environmental characteristics.\n")
        
        return str(report_path)

    def run_comprehensive_analysis(self) -> ComparisonResult:
        """Run the complete comprehensive analysis"""
        print("Starting Advanced Algorithm Analysis...")
        
        # Load all algorithm data
        algorithm_data = self.load_all_algorithm_data()
        
        if not algorithm_data:
            print("No algorithm data found. Please run algorithms first.")
            return None
        
        print(f"Loaded data for {len(algorithm_data)} algorithms")
        
        # Analyze each algorithm
        analyses = self.analyze_algorithm_performance(algorithm_data)
        print(f"Completed analysis for {len(analyses)} algorithms")
        
        # Generate visualizations
        print("Generating comprehensive visualizations...")
        visualizations = self.generate_comprehensive_visualizations(analyses)
        print(f"Generated {len(visualizations)} visualization files")
        
        # Generate report
        print("Generating comprehensive report...")
        report_path = self.generate_comprehensive_report(analyses)
        print(f"Report saved to: {report_path}")
        
        # Comparative statistics
        comparative_stats = self._generate_comparative_statistics(analyses)
        
        # Recommendations
        recommendations = self._generate_recommendations(analyses)
        
        result = ComparisonResult(
            algorithm_analyses=analyses,
            comparative_statistics=comparative_stats,
            recommendations=recommendations,
            visualizations_generated=visualizations
        )
        
        print("Advanced analysis completed successfully!")
        return result

    def _generate_comparative_statistics(self, analyses: List[AlgorithmAnalysis]) -> Dict[str, Any]:
        """Generate comparative statistics across all algorithms"""
        stats = {}
        
        # Performance statistics
        all_steps = [a.performance_metrics.get('avg_steps', 0) for a in analyses]
        all_times = [a.performance_metrics.get('avg_time', 0) for a in analyses]
        all_rewards = [a.performance_metrics.get('avg_reward', 0) for a in analyses]
        
        stats['performance'] = {
            'steps_mean': np.mean(all_steps),
            'steps_std': np.std(all_steps),
            'time_mean': np.mean(all_times),
            'time_std': np.std(all_times),
            'reward_mean': np.mean(all_rewards),
            'reward_std': np.std(all_rewards)
        }
        
        # Category statistics
        category_performance = {}
        for analysis in analyses:
            category = analysis.category
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(analysis.performance_metrics.get('avg_steps', 0))
        
        stats['category_performance'] = {
            cat: {
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }
            for cat, values in category_performance.items()
        }
        
        return stats

    def _generate_recommendations(self, analyses: List[AlgorithmAnalysis]) -> Dict[str, str]:
        """Generate specific recommendations based on analysis"""
        recommendations = {}
        
        # Find best performers
        best_overall = min(analyses, key=lambda a: a.performance_metrics.get('avg_steps', float('inf')))
        fastest = min(analyses, key=lambda a: a.performance_metrics.get('avg_time', float('inf')))
        most_efficient = max(analyses, key=lambda a: a.performance_metrics.get('efficiency', 0))
        
        recommendations['best_overall'] = f"{best_overall.name} - Best overall performance with {best_overall.performance_metrics.get('avg_steps', 0):.1f} average steps"
        recommendations['fastest'] = f"{fastest.name} - Fastest execution with {fastest.performance_metrics.get('avg_time', 0):.2f} average seconds"
        recommendations['most_efficient'] = f"{most_efficient.name} - Most efficient with {most_efficient.performance_metrics.get('efficiency', 0):.2f} steps/second"
        
        # Scenario-specific recommendations
        recommendations['real_time'] = "epsilon_greedy variants - Low computational overhead for real-time decisions"
        recommendations['optimal_solution'] = "astar or bfs - Guaranteed optimal solutions"
        recommendations['learning'] = "ppo - Adaptive learning capabilities"
        recommendations['exploration'] = "epsilon_greedy_alta_exploracion or tabu_search - High exploration strategies"
        
        return recommendations

def main():
    """Main execution function"""
    analyzer = AdvancedAlgorithmAnalyzer()
    result = analyzer.run_comprehensive_analysis()
    
    if result:
        print("\n" + "="*60)
        print("ADVANCED ALGORITHM ANALYSIS COMPLETED")
        print("="*60)
        print(f"Algorithms analyzed: {len(result.algorithm_analyses)}")
        print(f"Visualizations generated: {len(result.visualizations_generated)}")
        print("Files created in 'advanced_comparison_results/' directory")
        print("="*60)

if __name__ == "__main__":
    main()