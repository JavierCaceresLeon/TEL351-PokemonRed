#!/usr/bin/env python3
"""
Enhanced Professional Algorithm Comparison System
================================================

Advanced comparison system for all implemented algorithms in Pokemon Red environment.
This system provides comprehensive analysis, beautiful visualizations, and detailed
performance metrics with professional-grade implementation quality.

Features:
- PyBoy compatible execution environment
- Advanced statistical analysis
- Beautiful, publication-ready visualizations
- Comprehensive performance profiling
- Professional code quality standards
- Detailed algorithm pros/cons analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import psutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("plasma")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

@dataclass
class AlgorithmProfile:
    """Professional algorithm profiling data structure"""
    name: str
    category: str
    complexity_time: str
    complexity_space: str
    strengths: List[str]
    weaknesses: List[str]
    optimal_scenarios: List[str]
    implementation_quality: str
    pyboy_compatible: bool
    real_time_factor: float

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics structure"""
    algorithm: str
    execution_time: float
    total_steps: int
    total_reward: float
    memory_usage: float
    cpu_usage: float
    convergence_rate: float
    exploration_efficiency: float
    decision_quality: float
    stability_score: float

class EnhancedAlgorithmComparison:
    """
    Professional-grade algorithm comparison system
    """
    
    def __init__(self):
        self.results_dir = Path('RESULTADOS')
        self.output_dir = Path('enhanced_comparison_results')
        self.output_dir.mkdir(exist_ok=True)
        
        # Algorithm profiles with professional analysis
        self.algorithm_profiles = self._initialize_algorithm_profiles()
        
        # Performance tracking
        self.all_metrics = []
        self.comparison_data = pd.DataFrame()
        
        # Visualization settings
        self.color_schemes = {
            'search': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
            'learning': ['#8B5CF6', '#EC4899', '#10B981'],
            'combined': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#8B5CF6', '#EC4899', '#10B981', '#96CEB4', '#FECA57']
        }
    
    def _initialize_algorithm_profiles(self) -> Dict[str, AlgorithmProfile]:
        """Initialize comprehensive algorithm profiles"""
        return {
            'ppo': AlgorithmProfile(
                name='PPO (Proximal Policy Optimization)',
                category='Deep Reinforcement Learning',
                complexity_time='O(n×m×k) per episode',
                complexity_space='O(n×m) network parameters',
                strengths=[
                    'Excellent long-term learning capability',
                    'Robust policy optimization',
                    'Handles complex state spaces well',
                    'Proven performance in similar environments'
                ],
                weaknesses=[
                    'Requires extensive training time',
                    'High computational requirements',
                    'Black-box decision making',
                    'Needs large datasets for optimal performance'
                ],
                optimal_scenarios=[
                    'Complex decision making with delayed rewards',
                    'Environments with large state spaces',
                    'Long-term strategic planning',
                    'When sample efficiency is not critical'
                ],
                implementation_quality='Production-Ready',
                pyboy_compatible=True,
                real_time_factor=1.0
            ),
            'epsilon_greedy': AlgorithmProfile(
                name='Epsilon-Greedy with Heuristics',
                category='Simple Reinforcement Learning',
                complexity_time='O(1) per decision',
                complexity_space='O(k) heuristic parameters',
                strengths=[
                    'Extremely fast decision making',
                    'Transparent and interpretable',
                    'Easy to debug and modify',
                    'Low computational overhead'
                ],
                weaknesses=[
                    'Limited learning capability',
                    'Relies heavily on heuristic quality',
                    'May get stuck in local optima',
                    'Poor performance in complex scenarios'
                ],
                optimal_scenarios=[
                    'Real-time decision making requirements',
                    'Simple exploration tasks',
                    'When interpretability is crucial',
                    'Resource-constrained environments'
                ],
                implementation_quality='Baseline',
                pyboy_compatible=True,
                real_time_factor=6.0
            ),
            'astar': AlgorithmProfile(
                name='A* Search',
                category='Informed Search',
                complexity_time='O(b^d) worst case',
                complexity_space='O(b^d) nodes in memory',
                strengths=[
                    'Guaranteed optimal solution',
                    'Efficient with good heuristics',
                    'Complete and optimal',
                    'Well-understood theoretical properties'
                ],
                weaknesses=[
                    'High memory requirements',
                    'Heuristic quality dependent',
                    'May be slow in large spaces',
                    'Requires complete state representation'
                ],
                optimal_scenarios=[
                    'Pathfinding and navigation',
                    'When optimal solution is required',
                    'Static or slowly changing environments',
                    'Well-defined goal states'
                ],
                implementation_quality='Research-Grade',
                pyboy_compatible=True,
                real_time_factor=4.0
            ),
            'bfs': AlgorithmProfile(
                name='Breadth-First Search',
                category='Uninformed Search',
                complexity_time='O(b^d)',
                complexity_space='O(b^d)',
                strengths=[
                    'Guaranteed shortest path',
                    'Complete algorithm',
                    'Systematic exploration',
                    'Simple implementation'
                ],
                weaknesses=[
                    'Exponential space complexity',
                    'Very slow for deep solutions',
                    'No heuristic guidance',
                    'Impractical for large spaces'
                ],
                optimal_scenarios=[
                    'Small search spaces',
                    'When shortest path is critical',
                    'Educational/baseline comparisons',
                    'Graph problems with uniform costs'
                ],
                implementation_quality='Educational',
                pyboy_compatible=True,
                real_time_factor=8.0
            ),
            'tabu_search': AlgorithmProfile(
                name='Tabu Search',
                category='Metaheuristic Search',
                complexity_time='O(k×n) per iteration',
                complexity_space='O(t) tabu list size',
                strengths=[
                    'Escapes local optima effectively',
                    'Memory-based guidance',
                    'Good for large search spaces',
                    'Adaptive exploration strategy'
                ],
                weaknesses=[
                    'Parameter tuning required',
                    'No optimality guarantees',
                    'Tabu list management overhead',
                    'May cycle in pathological cases'
                ],
                optimal_scenarios=[
                    'Large search spaces',
                    'When local optima are problematic',
                    'Combinatorial optimization',
                    'Long-term exploration needs'
                ],
                implementation_quality='Advanced',
                pyboy_compatible=True,
                real_time_factor=5.0
            ),
            'hill_climbing': AlgorithmProfile(
                name='Hill Climbing (Multiple Variants)',
                category='Local Search',
                complexity_time='O(n) per iteration',
                complexity_space='O(1) current state',
                strengths=[
                    'Very fast execution',
                    'Low memory requirements',
                    'Simple implementation',
                    'Good for local optimization'
                ],
                weaknesses=[
                    'Gets stuck in local optima',
                    'No backtracking capability',
                    'Sensitive to initial state',
                    'Poor in plateau regions'
                ],
                optimal_scenarios=[
                    'Quick approximation needs',
                    'Local optimization problems',
                    'Resource-constrained scenarios',
                    'When restarts are possible'
                ],
                implementation_quality='Basic',
                pyboy_compatible=True,
                real_time_factor=6.0
            ),
            'simulated_annealing': AlgorithmProfile(
                name='Simulated Annealing',
                category='Stochastic Search',
                complexity_time='O(k×n) iterations',
                complexity_space='O(1) current state',
                strengths=[
                    'Escapes local optima probabilistically',
                    'Good global optimization properties',
                    'Annealing schedule flexibility',
                    'Works well in large spaces'
                ],
                weaknesses=[
                    'Parameter tuning critical',
                    'No optimality guarantees',
                    'Cooling schedule dependency',
                    'May be slow to converge'
                ],
                optimal_scenarios=[
                    'Global optimization needs',
                    'Large search spaces',
                    'When good approximation is sufficient',
                    'Combinatorial problems'
                ],
                implementation_quality='Intermediate',
                pyboy_compatible=True,
                real_time_factor=7.0
            )
        }
    
    def load_comprehensive_data(self) -> pd.DataFrame:
        """Load and process all available experimental data"""
        print("Loading comprehensive experimental data...")
        
        all_data = []
        summary_files = list(self.results_dir.rglob('*summary*.csv'))
        
        print(f"Found {len(summary_files)} summary files")
        
        for file_path in summary_files:
            try:
                df = pd.read_csv(file_path)
                
                # Pivot data to get metrics as columns
                if 'Métrica' in df.columns and 'Valor' in df.columns:
                    pivoted = df.pivot_table(index=None, columns='Métrica', values='Valor', aggfunc='first')
                    
                    # Get algorithm name from path
                    algorithm = self._extract_algorithm_name(file_path)
                    pivoted['Algorithm'] = algorithm
                    pivoted['Source_File'] = str(file_path)
                    
                    all_data.append(pivoted)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not all_data:
            print("No valid data found!")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Standardize column names
        if 'Pasos Totales' in combined_df.columns:
            combined_df['Total_Steps'] = combined_df['Pasos Totales']
        if 'Tiempo (s)' in combined_df.columns:
            combined_df['Execution_Time'] = combined_df['Tiempo (s)']
        if 'Recompensa Total' in combined_df.columns:
            combined_df['Total_Reward'] = combined_df['Recompensa Total']
        
        # Clean up data
        combined_df = combined_df.dropna(subset=['Total_Steps', 'Execution_Time'])
        
        # Convert to numeric, replacing any non-numeric values with NaN
        numeric_columns = ['Total_Steps', 'Execution_Time', 'Total_Reward']
        for col in numeric_columns:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Drop rows with any NaN values in essential columns
        combined_df = combined_df.dropna(subset=numeric_columns)
        
        # Calculate derived metrics with safe operations
        combined_df['Efficiency'] = combined_df['Total_Reward'] / combined_df['Total_Steps'].clip(lower=1.0)
        combined_df['Speed'] = combined_df['Total_Steps'] / combined_df['Execution_Time'].clip(lower=0.1)
        
        print(f"Processed {len(combined_df)} experimental runs across {combined_df['Algorithm'].nunique()} algorithms")
        
        return combined_df
    
    def _extract_algorithm_name(self, file_path: Path) -> str:
        """Extract clean algorithm name from file path"""
        parts = file_path.parts
        
        # Algorithm mapping based on directory structure
        if 'ppo' in parts:
            return 'ppo'
        elif 'epsilon_greedy_comparison' in parts:
            config_part = [p for p in parts if p in ['alta_exploracion', 'moderada_alta', 'balanceada', 'conservadora', 'muy_greedy']]
            if config_part:
                return f'epsilon_greedy_{config_part[0]}'
            return 'epsilon_greedy'
        elif 'search_algorithms_comparison' in parts:
            algo_part = [p for p in parts if any(algo in p for algo in ['astar', 'bfs', 'tabu', 'hill', 'simulated'])]
            if algo_part:
                return algo_part[0]
        elif 'epsilon_greedy' in parts:
            return 'epsilon_greedy'
        
        return 'unknown'
    
    def generate_professional_visualizations(self, df: pd.DataFrame):
        """Generate professional-grade visualizations"""
        print("Generating professional visualizations...")
        
        # 1. Comprehensive Performance Comparison
        self._create_performance_dashboard(df)
        
        # 2. Algorithm Category Analysis
        self._create_category_analysis(df)
        
        # 3. Statistical Distribution Analysis
        self._create_distribution_analysis(df)
        
        # 4. Efficiency vs Speed Analysis
        self._create_efficiency_analysis(df)
        
        # 5. Algorithm Recommendation Matrix
        self._create_recommendation_matrix(df)
        
        print("All visualizations generated successfully!")
    
    def _get_color_mapping(self, algorithms: List[str]) -> Dict[str, str]:
        """Create robust color mapping for algorithms"""
        base_colors = self.color_schemes['combined']
        colors = {}
        for i, alg in enumerate(algorithms):
            colors[alg] = base_colors[i % len(base_colors)]
        return colors
    
    def _create_performance_dashboard(self, df: pd.DataFrame):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # Color mapping for algorithms with fallback
        algorithms = df['Algorithm'].unique()
        colors = self._get_color_mapping(algorithms)
        
        # 1. Execution Time Comparison (Log Scale)
        ax1 = fig.add_subplot(gs[0, 0])
        df_grouped = df.groupby('Algorithm')['Execution_Time'].agg(['mean', 'std']).reset_index()
        bars = ax1.bar(df_grouped['Algorithm'], df_grouped['mean'], 
                      yerr=df_grouped['std'], capsize=5,
                      color=[colors[alg] for alg in df_grouped['Algorithm']],
                      alpha=0.8)
        ax1.set_yscale('log')
        ax1.set_title('Execution Time Comparison\n(Log Scale)', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Total Steps Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        df_grouped_steps = df.groupby('Algorithm')['Total_Steps'].agg(['mean', 'std']).reset_index()
        bars = ax2.bar(df_grouped_steps['Algorithm'], df_grouped_steps['mean'],
                      yerr=df_grouped_steps['std'], capsize=5,
                      color=[colors[alg] for alg in df_grouped_steps['Algorithm']],
                      alpha=0.8)
        ax2.set_title('Total Steps Comparison', fontweight='bold')
        ax2.set_ylabel('Steps')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Efficiency Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        df_grouped_eff = df.groupby('Algorithm')['Efficiency'].agg(['mean', 'std']).reset_index()
        bars = ax3.bar(df_grouped_eff['Algorithm'], df_grouped_eff['mean'],
                      yerr=df_grouped_eff['std'], capsize=5,
                      color=[colors[alg] for alg in df_grouped_eff['Algorithm']],
                      alpha=0.8)
        ax3.set_title('Reward Efficiency\n(Reward/Step)', fontweight='bold')
        ax3.set_ylabel('Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Violin Plot - Time Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        df_plot = df[df['Execution_Time'] < df['Execution_Time'].quantile(0.95)]  # Remove outliers
        sns.violinplot(data=df_plot, x='Algorithm', y='Execution_Time', ax=ax4)
        ax4.set_title('Time Distribution\n(Violin Plot)', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Box Plot - Steps Distribution
        ax5 = fig.add_subplot(gs[1, 1])
        df_plot_steps = df[df['Total_Steps'] < df['Total_Steps'].quantile(0.95)]
        sns.boxplot(data=df_plot_steps, x='Algorithm', y='Total_Steps', ax=ax5)
        ax5.set_title('Steps Distribution\n(Box Plot)', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Scatter Plot - Efficiency vs Speed
        ax6 = fig.add_subplot(gs[1, 2])
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            ax6.scatter(alg_data['Speed'], alg_data['Efficiency'], 
                       color=colors[alg], label=alg, alpha=0.7, s=50)
        ax6.set_xlabel('Speed (Steps/Second)')
        ax6.set_ylabel('Efficiency (Reward/Step)')
        ax6.set_title('Speed vs Efficiency', fontweight='bold')
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 7. Radar Chart - Multi-dimensional Comparison
        ax7 = fig.add_subplot(gs[2, :], projection='polar')
        self._create_radar_chart(df, ax7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_performance_dashboard.png')
        plt.close()
    
    def _create_radar_chart(self, df: pd.DataFrame, ax):
        """Create radar chart for multi-dimensional comparison"""
        algorithms = df['Algorithm'].unique()[:6]  # Limit for readability
        
        # Metrics for radar chart
        metrics = ['Execution_Time', 'Total_Steps', 'Efficiency', 'Speed']
        metric_labels = ['Speed\n(1/Time)', 'Compactness\n(1/Steps)', 'Efficiency', 'Execution\nSpeed']
        
        # Normalize metrics (invert time and steps for better visualization)
        normalized_data = {}
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            values = []
            values.append(1 / alg_data['Execution_Time'].mean())  # Speed (inverse of time)
            values.append(1 / alg_data['Total_Steps'].mean())     # Compactness (inverse of steps)
            values.append(alg_data['Efficiency'].mean())          # Efficiency
            values.append(alg_data['Speed'].mean())               # Execution speed
            
            # Normalize to 0-1 scale
            normalized_data[alg] = [(v - min(values)) / (max(values) - min(values) + 1e-8) for v in values]
        
        # Angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each algorithm
        colors = self.color_schemes['combined'][:len(algorithms)]
        for i, alg in enumerate(algorithms):
            values = normalized_data[alg] + normalized_data[alg][:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Dimensional Algorithm Comparison\n(Normalized Metrics)', fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def _create_category_analysis(self, df: pd.DataFrame):
        """Create algorithm category analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Add category information to dataframe
        df_cat = df.copy()
        df_cat['Category'] = df_cat['Algorithm'].map(
            lambda x: self.algorithm_profiles.get(x.split('_')[0], 
                                                 AlgorithmProfile('Unknown', 'Unknown', '', '', [], [], [], '', False, 1.0)).category
        )
        
        # 1. Performance by Category
        ax1 = axes[0, 0]
        category_performance = df_cat.groupby('Category')['Efficiency'].agg(['mean', 'std']).reset_index()
        bars = ax1.bar(category_performance['Category'], category_performance['mean'],
                      yerr=category_performance['std'], capsize=5, alpha=0.8)
        ax1.set_title('Average Efficiency by Algorithm Category', fontweight='bold')
        ax1.set_ylabel('Efficiency (Reward/Step)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Execution Time by Category
        ax2 = axes[0, 1]
        sns.boxplot(data=df_cat, x='Category', y='Execution_Time', ax=ax2)
        ax2.set_yscale('log')
        ax2.set_title('Execution Time Distribution by Category', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Steps vs Time Scatter by Category
        ax3 = axes[1, 0]
        for category in df_cat['Category'].unique():
            cat_data = df_cat[df_cat['Category'] == category]
            ax3.scatter(cat_data['Execution_Time'], cat_data['Total_Steps'], 
                       label=category, alpha=0.7, s=50)
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_ylabel('Total Steps')
        ax3.set_title('Steps vs Time by Algorithm Category', fontweight='bold')
        ax3.legend()
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # 4. Category Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate category statistics
        cat_stats = df_cat.groupby('Category').agg({
            'Execution_Time': ['mean', 'std'],
            'Total_Steps': ['mean', 'std'],
            'Efficiency': ['mean', 'std']
        }).round(3)
        
        # Create table
        table_data = []
        for category in cat_stats.index:
            row = [
                category,
                f"{cat_stats.loc[category, ('Execution_Time', 'mean')]:.2f}±{cat_stats.loc[category, ('Execution_Time', 'std')]:.2f}",
                f"{cat_stats.loc[category, ('Total_Steps', 'mean')]:.1f}±{cat_stats.loc[category, ('Total_Steps', 'std')]:.1f}",
                f"{cat_stats.loc[category, ('Efficiency', 'mean')]:.3f}±{cat_stats.loc[category, ('Efficiency', 'std')]:.3f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Category', 'Time (s)', 'Steps', 'Efficiency'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Category Performance Summary', fontweight='bold', y=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_analysis.png')
        plt.close()
    
    def _create_distribution_analysis(self, df: pd.DataFrame):
        """Create statistical distribution analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        algorithms = df['Algorithm'].unique()
        colors = self._get_color_mapping(algorithms)
        
        # 1. Time Distribution Histogram
        ax1 = axes[0, 0]
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]['Execution_Time']
            ax1.hist(alg_data, alpha=0.6, label=alg, bins=15, color=colors[alg])
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Execution Time Distribution', fontweight='bold')
        ax1.legend()
        ax1.set_yscale('log')
        
        # 2. Steps Distribution Histogram
        ax2 = axes[0, 1]
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]['Total_Steps']
            ax2.hist(alg_data, alpha=0.6, label=alg, bins=15, color=colors[alg])
        ax2.set_xlabel('Total Steps')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Steps Distribution', fontweight='bold')
        ax2.legend()
        
        # 3. Efficiency Distribution
        ax3 = axes[0, 2]
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]['Efficiency']
            ax3.hist(alg_data, alpha=0.6, label=alg, bins=15, color=colors[alg])
        ax3.set_xlabel('Efficiency (Reward/Step)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Efficiency Distribution', fontweight='bold')
        ax3.legend()
        
        # 4. Statistical Test Results
        ax4 = axes[1, 0]
        ax4.axis('off')
        
        # Perform statistical tests
        time_groups = [df[df['Algorithm'] == alg]['Execution_Time'].values for alg in algorithms]
        f_stat, p_value = stats.f_oneway(*time_groups)
        
        test_results = [
            f"ANOVA Test - Execution Time:",
            f"F-statistic: {f_stat:.4f}",
            f"P-value: {p_value:.4e}",
            "",
            f"Interpretation:",
            f"{'Significant' if p_value < 0.05 else 'Not significant'} difference",
            f"between algorithm performance",
            f"(α = 0.05)"
        ]
        
        for i, text in enumerate(test_results):
            ax4.text(0.1, 0.9 - i*0.1, text, transform=ax4.transAxes, fontsize=11,
                    fontweight='bold' if i == 0 or i == 4 else 'normal')
        
        # 5. Correlation Matrix
        ax5 = axes[1, 1]
        corr_data = df[['Execution_Time', 'Total_Steps', 'Efficiency', 'Speed']].corr()
        im = ax5.imshow(corr_data.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(corr_data.columns)))
        ax5.set_yticks(range(len(corr_data.columns)))
        ax5.set_xticklabels(['Time', 'Steps', 'Efficiency', 'Speed'])
        ax5.set_yticklabels(['Time', 'Steps', 'Efficiency', 'Speed'])
        ax5.set_title('Metric Correlation Matrix', fontweight='bold')
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                ax5.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white' if abs(corr_data.iloc[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # 6. Performance Ranking
        ax6 = axes[1, 2]
        
        # Calculate composite score
        df_rank = df.groupby('Algorithm').agg({
            'Execution_Time': 'mean',
            'Total_Steps': 'mean', 
            'Efficiency': 'mean'
        }).reset_index()
        
        # Normalize and create composite score (lower time and steps are better)
        scaler = StandardScaler()
        df_rank['Time_Score'] = -scaler.fit_transform(df_rank[['Execution_Time']])[:,0]
        df_rank['Steps_Score'] = -scaler.fit_transform(df_rank[['Total_Steps']])[:,0]
        df_rank['Efficiency_Score'] = scaler.fit_transform(df_rank[['Efficiency']])[:,0]
        
        df_rank['Composite_Score'] = (df_rank['Time_Score'] + df_rank['Steps_Score'] + df_rank['Efficiency_Score']) / 3
        df_rank = df_rank.sort_values('Composite_Score', ascending=False)
        
        bars = ax6.barh(df_rank['Algorithm'], df_rank['Composite_Score'],
                       color=[colors[alg] for alg in df_rank['Algorithm']])
        ax6.set_xlabel('Composite Performance Score')
        ax6.set_title('Overall Algorithm Ranking', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_analysis.png')
        plt.close()
    
    def _create_efficiency_analysis(self, df: pd.DataFrame):
        """Create efficiency analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Efficiency vs Time Trade-off
        ax1 = axes[0, 0]
        algorithms = df['Algorithm'].unique()
        colors = self._get_color_mapping(algorithms)
        
        for alg in algorithms:
            alg_data = df[df['Algorithm'] == alg]
            ax1.scatter(alg_data['Execution_Time'], alg_data['Efficiency'], 
                       color=colors[alg], label=alg, alpha=0.7, s=50)
        
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_ylabel('Efficiency (Reward/Step)')
        ax1.set_title('Efficiency vs Execution Time Trade-off', fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pareto Frontier Analysis
        ax2 = axes[0, 1]
        
        # Calculate algorithm averages for Pareto analysis
        algo_means = df.groupby('Algorithm').agg({
            'Execution_Time': 'mean',
            'Efficiency': 'mean'
        }).reset_index()
        
        # Plot Pareto frontier
        sorted_algos = algo_means.sort_values('Execution_Time')
        pareto_efficient = []
        max_efficiency = -float('inf')
        
        for _, row in sorted_algos.iterrows():
            if row['Efficiency'] > max_efficiency:
                pareto_efficient.append(row)
                max_efficiency = row['Efficiency']
        
        # Plot all algorithms
        for _, row in algo_means.iterrows():
            color = colors[row['Algorithm']]
            marker = 'o' if any(p['Algorithm'] == row['Algorithm'] for p in pareto_efficient) else 's'
            size = 100 if any(p['Algorithm'] == row['Algorithm'] for p in pareto_efficient) else 50
            ax2.scatter(row['Execution_Time'], row['Efficiency'], 
                       color=color, marker=marker, s=size, label=row['Algorithm'])
        
        # Draw Pareto frontier
        if len(pareto_efficient) > 1:
            pareto_df = pd.DataFrame(pareto_efficient)
            ax2.plot(pareto_df['Execution_Time'], pareto_df['Efficiency'], 
                    'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
        
        ax2.set_xlabel('Execution Time (seconds)')
        ax2.set_ylabel('Efficiency (Reward/Step)')
        ax2.set_title('Pareto Efficiency Analysis', fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Robustness Analysis (Coefficient of Variation)
        ax3 = axes[1, 0]
        
        robustness_data = df.groupby('Algorithm').agg({
            'Execution_Time': ['mean', 'std'],
            'Efficiency': ['mean', 'std']
        })
        
        # Calculate coefficient of variation
        time_cv = robustness_data[('Execution_Time', 'std')] / robustness_data[('Execution_Time', 'mean')]
        efficiency_cv = robustness_data[('Efficiency', 'std')] / robustness_data[('Efficiency', 'mean')]
        
        for alg in algorithms:
            ax3.scatter(time_cv[alg], efficiency_cv[alg], 
                       color=colors[alg], label=alg, s=100)
        
        ax3.set_xlabel('Time Variability (CV)')
        ax3.set_ylabel('Efficiency Variability (CV)')
        ax3.set_title('Algorithm Robustness Analysis', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Stability Over Time
        ax4 = axes[1, 1]
        
        # If we have run order information, plot stability
        # For now, plot efficiency distributions as box plots
        efficiency_data = []
        algorithm_labels = []
        
        for alg in algorithms:
            alg_efficiencies = df[df['Algorithm'] == alg]['Efficiency'].tolist()
            efficiency_data.append(alg_efficiencies)
            algorithm_labels.append(alg)
        
        bp = ax4.boxplot(efficiency_data, labels=algorithm_labels, patch_artist=True)
        
        # Color the boxes
        for patch, alg in zip(bp['boxes'], algorithm_labels):
            patch.set_facecolor(colors[alg])
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Efficiency (Reward/Step)')
        ax4.set_title('Efficiency Stability Analysis', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_analysis.png')
        plt.close()
    
    def _create_recommendation_matrix(self, df: pd.DataFrame):
        """Create algorithm recommendation matrix based on scenarios"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define scenarios and their weights
        scenarios = {
            'Real-time Performance': {'time_weight': 0.8, 'efficiency_weight': 0.2},
            'Maximum Efficiency': {'time_weight': 0.2, 'efficiency_weight': 0.8},
            'Balanced Performance': {'time_weight': 0.5, 'efficiency_weight': 0.5},
            'Resource Constrained': {'time_weight': 0.6, 'efficiency_weight': 0.4}
        }
        
        # Calculate scenario scores
        algo_means = df.groupby('Algorithm').agg({
            'Execution_Time': 'mean',
            'Efficiency': 'mean'
        }).reset_index()
        
        # Normalize metrics (lower time is better, higher efficiency is better)
        max_time = algo_means['Execution_Time'].max()
        min_time = algo_means['Execution_Time'].min()
        max_eff = algo_means['Efficiency'].max()
        min_eff = algo_means['Efficiency'].min()
        
        algo_means['Time_Score'] = 1 - (algo_means['Execution_Time'] - min_time) / (max_time - min_time)
        algo_means['Efficiency_Score'] = (algo_means['Efficiency'] - min_eff) / (max_eff - min_eff)
        
        # Calculate scenario scores
        scenario_scores = {}
        for scenario, weights in scenarios.items():
            algo_means[f'{scenario}_Score'] = (
                weights['time_weight'] * algo_means['Time_Score'] + 
                weights['efficiency_weight'] * algo_means['Efficiency_Score']
            )
            scenario_scores[scenario] = algo_means[['Algorithm', f'{scenario}_Score']].set_index('Algorithm')[f'{scenario}_Score'].to_dict()
        
        # 1. Recommendation Heatmap
        ax1 = axes[0, 0]
        
        recommendation_matrix = pd.DataFrame(scenario_scores).T
        im = ax1.imshow(recommendation_matrix.values, cmap='RdYlGn', aspect='auto')
        
        ax1.set_xticks(range(len(recommendation_matrix.columns)))
        ax1.set_yticks(range(len(recommendation_matrix.index)))
        ax1.set_xticklabels(recommendation_matrix.columns, rotation=45, ha='right')
        ax1.set_yticklabels(recommendation_matrix.index)
        ax1.set_title('Algorithm Recommendation Matrix', fontweight='bold')
        
        # Add score values
        for i in range(len(recommendation_matrix.index)):
            for j in range(len(recommendation_matrix.columns)):
                ax1.text(j, i, f'{recommendation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', color='white' if recommendation_matrix.iloc[i, j] < 0.5 else 'black')
        
        plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # 2. Best Algorithm per Scenario
        ax2 = axes[0, 1]
        ax2.axis('off')
        
        recommendations = []
        for scenario in scenarios:
            best_algo = max(scenario_scores[scenario], key=scenario_scores[scenario].get)
            best_score = scenario_scores[scenario][best_algo]
            recommendations.append([scenario, best_algo, f'{best_score:.3f}'])
        
        table = ax2.table(cellText=recommendations,
                         colLabels=['Scenario', 'Best Algorithm', 'Score'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)
        ax2.set_title('Scenario-Based Recommendations', fontweight='bold', y=0.8)
        
        # 3. Algorithm Complexity vs Performance
        ax3 = axes[1, 0]
        
        # Create complexity scoring (simplified)
        complexity_scores = {
            'ppo': 5,  # Highest complexity
            'epsilon_greedy': 1,  # Lowest complexity
            'astar': 4,
            'bfs': 2,
            'tabu_search': 4,
            'hill_climbing': 2,
            'simulated_annealing': 3
        }
        
        for alg in algo_means['Algorithm']:
            if any(key in alg for key in complexity_scores.keys()):
                base_alg = next(key for key in complexity_scores.keys() if key in alg)
                complexity = complexity_scores[base_alg]
                efficiency = algo_means[algo_means['Algorithm'] == alg]['Efficiency'].iloc[0]
                
                ax3.scatter(complexity, efficiency, s=100, alpha=0.7, label=alg)
                ax3.annotate(alg, (complexity, efficiency), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Implementation Complexity (1=Simple, 5=Complex)')
        ax3.set_ylabel('Efficiency (Reward/Step)')
        ax3.set_title('Complexity vs Performance Trade-off', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Use Case Decision Tree
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        decision_tree = [
            "ALGORITHM SELECTION DECISION TREE",
            "",
            "1. Real-time requirement?",
            "   YES → Epsilon-Greedy or Hill Climbing",
            "   NO → Continue to 2",
            "",
            "2. Optimal solution required?", 
            "   YES → A* Search (if feasible)",
            "   NO → Continue to 3",
            "",
            "3. Complex learning needed?",
            "   YES → PPO",
            "   NO → Continue to 4", 
            "",
            "4. Large search space?",
            "   YES → Tabu Search or Simulated Annealing",
            "   NO → BFS or Hill Climbing"
        ]
        
        for i, text in enumerate(decision_tree):
            weight = 'bold' if i == 0 or text.startswith(('1.', '2.', '3.', '4.')) else 'normal'
            ax4.text(0.05, 0.95 - i*0.05, text, transform=ax4.transAxes, 
                    fontsize=10, fontweight=weight)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recommendation_matrix.png')
        plt.close()
    
    def generate_comprehensive_report(self, df: pd.DataFrame):
        """Generate comprehensive LaTeX report"""
        print("Generating comprehensive analysis report...")
        
        # Calculate summary statistics
        summary_stats = df.groupby('Algorithm').agg({
            'Execution_Time': ['mean', 'std', 'min', 'max'],
            'Total_Steps': ['mean', 'std', 'min', 'max'],
            'Efficiency': ['mean', 'std', 'min', 'max'],
            'Speed': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Generate algorithm analysis
        algorithm_analysis = self._generate_algorithm_analysis(df)
        
        # Create comprehensive LaTeX table
        latex_table = self._create_enhanced_latex_table(summary_stats)
        
        # Save results
        with open(self.output_dir / 'enhanced_comparison_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        with open(self.output_dir / 'algorithm_analysis.tex', 'w', encoding='utf-8') as f:
            f.write(algorithm_analysis)
        
        print("Comprehensive report generated successfully!")
    
    def _generate_algorithm_analysis(self, df: pd.DataFrame) -> str:
        """Generate detailed algorithm analysis in LaTeX format"""
        analysis_sections = []
        
        for algo_key, profile in self.algorithm_profiles.items():
            # Check if we have data for this algorithm
            algo_data = df[df['Algorithm'].str.contains(algo_key)]
            if algo_data.empty:
                continue
            
            # Calculate statistics
            avg_time = algo_data['Execution_Time'].mean()
            avg_steps = algo_data['Total_Steps'].mean()
            avg_efficiency = algo_data['Efficiency'].mean()
            
            section = f"""
\\subsection{{{profile.name}}}

\\textbf{{Categoría:}} {profile.category}

\\textbf{{Complejidad Temporal:}} {profile.complexity_time}

\\textbf{{Complejidad Espacial:}} {profile.complexity_space}

\\textbf{{Rendimiento Experimental:}}
\\begin{{itemize}}
    \\item Tiempo promedio de ejecución: {avg_time:.2f} segundos
    \\item Pasos promedio: {avg_steps:.1f}
    \\item Eficiencia promedio: {avg_efficiency:.4f} recompensa/paso
\\end{{itemize}}

\\textbf{{Fortalezas:}}
\\begin{{itemize}}
"""
            for strength in profile.strengths:
                section += f"    \\item {strength}\n"
            
            section += "\\end{itemize}\n\n\\textbf{Debilidades:}\n\\begin{itemize}\n"
            
            for weakness in profile.weaknesses:
                section += f"    \\item {weakness}\n"
            
            section += "\\end{itemize}\n\n\\textbf{Escenarios Óptimos:}\n\\begin{itemize}\n"
            
            for scenario in profile.optimal_scenarios:
                section += f"    \\item {scenario}\n"
            
            section += f"""\\end{{itemize}}

\\textbf{{Calidad de Implementación:}} {profile.implementation_quality}

\\textbf{{Compatibilidad PyBoy:}} {'Sí' if profile.pyboy_compatible else 'No'}

\\textbf{{Factor de Tiempo Real:}} {profile.real_time_factor}x

"""
            
            analysis_sections.append(section)
        
        return "\\section{Análisis Detallado de Algoritmos}\n\n" + "\n".join(analysis_sections)
    
    def _create_enhanced_latex_table(self, summary_stats: pd.DataFrame) -> str:
        """Create enhanced LaTeX table with all metrics"""
        
        table_data = []
        for algo in summary_stats.index:
            row = [
                algo.replace('_', '\\_'),  # Escape underscores for LaTeX
                f"{summary_stats.loc[algo, ('Execution_Time', 'mean')]:.2f} ± {summary_stats.loc[algo, ('Execution_Time', 'std')]:.2f}",
                f"{summary_stats.loc[algo, ('Total_Steps', 'mean')]:.1f} ± {summary_stats.loc[algo, ('Total_Steps', 'std')]:.1f}",
                f"{summary_stats.loc[algo, ('Efficiency', 'mean')]:.4f} ± {summary_stats.loc[algo, ('Efficiency', 'std')]:.4f}",
                f"{summary_stats.loc[algo, ('Speed', 'mean')]:.1f} ± {summary_stats.loc[algo, ('Speed', 'std')]:.1f}"
            ]
            table_data.append(row)
        
        # Create LaTeX table
        latex_table = """\\begin{table}[H]
\\centering
\\caption{Comparación Exhaustiva de Algoritmos de Navegación en Pokémon Red}
\\label{tab:enhanced_algorithm_comparison}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Algoritmo} & \\textbf{Tiempo (s)} & \\textbf{Pasos Totales} & \\textbf{Eficiencia} & \\textbf{Velocidad (pasos/s)} \\\\
\\hline
"""
        
        for row in table_data:
            latex_table += " & ".join(row) + " \\\\\n\\hline\n"
        
        latex_table += """\\end{tabular}
}
\\end{table}

\\begin{table}[H]
\\centering
\\caption{Matriz de Recomendaciones por Escenario de Uso}
\\label{tab:algorithm_recommendations}
\\begin{tabular}{|l|l|l|}
\\hline
\\textbf{Escenario} & \\textbf{Algoritmo Recomendado} & \\textbf{Justificación} \\\\
\\hline
Tiempo Real & Epsilon-Greedy & Decisiones instantáneas, baja complejidad \\\\
\\hline
Precisión Óptima & A* Search & Garantiza solución óptima con heurística adecuada \\\\
\\hline
Aprendizaje Complejo & PPO & Capacidad de aprendizaje profundo y adaptación \\\\
\\hline
Espacios Grandes & Tabu Search & Evita óptimos locales, memoria adaptativa \\\\
\\hline
Recursos Limitados & Hill Climbing & Mínimos requisitos de memoria y procesamiento \\\\
\\hline
Exploración Equilibrada & Simulated Annealing & Balance exploración-explotación probabilístico \\\\
\\hline
\\end{tabular}
\\end{table}
"""
        
        return latex_table
    
    def run_complete_analysis(self):
        """Run the complete professional analysis"""
        print("Starting Enhanced Professional Algorithm Comparison")
        print("=" * 60)
        
        # Load data
        df = self.load_comprehensive_data()
        
        if df.empty:
            print("No data available for analysis!")
            return
        
        # Generate visualizations
        self.generate_professional_visualizations(df)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(df)
        
        # Save processed data
        df.to_csv(self.output_dir / 'processed_comparison_data.csv', index=False)
        
        print("\nAnalysis complete! Results saved to:", self.output_dir)
        print("\nGenerated files:")
        print("- comprehensive_performance_dashboard.png")
        print("- category_analysis.png") 
        print("- statistical_analysis.png")
        print("- efficiency_analysis.png")
        print("- recommendation_matrix.png")
        print("- enhanced_comparison_table.tex")
        print("- algorithm_analysis.tex")
        print("- processed_comparison_data.csv")

def main():
    """Main execution function"""
    analyzer = EnhancedAlgorithmComparison()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()