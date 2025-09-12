"""
Configuration Module for Pokemon Red Agent Comparison
====================================================

This module centralizes configuration settings for the agent comparison system.
"""

from pathlib import Path
from typing import Dict, Any
import json


class Config:
    """
    Configuration class for the Pokemon Red agent comparison
    """
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    V2_PATH = PROJECT_ROOT.parent / "v2"
    ASSETS_PATH = PROJECT_ROOT.parent / "assets"
    
    # Game files
    GB_ROM_PATH = PROJECT_ROOT.parent / "PokemonRed.gb"
    INIT_STATE_PATH = PROJECT_ROOT.parent / "init.state"
    
    # Default environment configuration
    DEFAULT_ENV_CONFIG = {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': str(INIT_STATE_PATH),
        'max_steps': 40960,
        'print_rewards': False,
        'save_video': False,
        'fast_video': True,
        'gb_path': str(GB_ROM_PATH),
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25
    }
    
    # Default comparison configuration
    DEFAULT_COMPARISON_CONFIG = {
        'num_episodes': 5,
        'max_steps_per_episode': 40960,
        'parallel_execution': False,
        'save_detailed_logs': True,
        'create_visualizations': True,
        'metrics_to_compare': [
            'episode_rewards',
            'episode_lengths', 
            'exploration_efficiency',
            'convergence_rate',
            'stability',
            'scenario_adaptation'
        ]
    }
    
    # Default Epsilon Greedy configuration
    DEFAULT_EPSILON_CONFIG = {
        'epsilon_start': 0.5,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.995,
        'scenario_detection_enabled': True
    }
    
    # Scenario-specific heuristic weights
    SCENARIO_WEIGHTS = {
        'exploration': {
            'exploration': 1.5,
            'objective_distance': 0.8,
            'health_consideration': 0.6,
            'level_progression': 0.4,
            'map_familiarity': 1.0,
            'event_completion': 0.8
        },
        'battle': {
            'exploration': 0.2,
            'objective_distance': 0.3,
            'health_consideration': 2.0,
            'level_progression': 1.5,
            'map_familiarity': 0.1,
            'event_completion': 0.5
        },
        'navigation': {
            'exploration': 0.6,
            'objective_distance': 2.0,
            'health_consideration': 0.8,
            'level_progression': 0.6,
            'map_familiarity': 1.2,
            'event_completion': 1.0
        },
        'progression': {
            'exploration': 0.8,
            'objective_distance': 1.0,
            'health_consideration': 1.0,
            'level_progression': 2.0,
            'map_familiarity': 0.8,
            'event_completion': 2.5
        },
        'stuck': {
            'exploration': 2.0,
            'objective_distance': 1.5,
            'health_consideration': 0.5,
            'level_progression': 0.8,
            'map_familiarity': 0.3,
            'event_completion': 1.0
        }
    }
    
    # Visualization settings
    VISUALIZATION_CONFIG = {
        'style': 'seaborn-v0_8',
        'figure_size': (15, 10),
        'dpi': 300,
        'color_palette': {
            'PPO': '#3498db',
            'Epsilon_Greedy': '#2ecc71',
            'Random': '#e74c3c',
            'Heuristic': '#f39c12'
        },
        'save_format': 'png'
    }
    
    # Metrics configuration
    METRICS_CONFIG = {
        'convergence_window': 5,
        'stability_threshold': 0.1,
        'exploration_efficiency_threshold': 0.5,
        'plateau_threshold': 0.05,
        'risk_free_rate': 0.0
    }
    
    # Output directories
    OUTPUT_DIRS = {
        'comparison_results': 'comparison_results',
        'metrics_analysis': 'metrics_analysis',
        'logs': 'logs',
        'models': 'models',
        'visualizations': 'visualizations'
    }
    
    @classmethod
    def get_env_config(cls, **overrides) -> Dict[str, Any]:
        """
        Get environment configuration with optional overrides
        """
        config = cls.DEFAULT_ENV_CONFIG.copy()
        config.update(overrides)
        return config
    
    @classmethod
    def get_comparison_config(cls, **overrides) -> Dict[str, Any]:
        """
        Get comparison configuration with optional overrides
        """
        config = cls.DEFAULT_COMPARISON_CONFIG.copy()
        config.update(overrides)
        return config
    
    @classmethod
    def get_epsilon_config(cls, **overrides) -> Dict[str, Any]:
        """
        Get Epsilon Greedy configuration with optional overrides
        """
        config = cls.DEFAULT_EPSILON_CONFIG.copy()
        config.update(overrides)
        return config
    
    @classmethod
    def get_scenario_weights(cls, scenario: str) -> Dict[str, float]:
        """
        Get heuristic weights for a specific scenario
        """
        return cls.SCENARIO_WEIGHTS.get(scenario, cls.SCENARIO_WEIGHTS['exploration'])
    
    @classmethod
    def setup_output_directories(cls) -> Dict[str, Path]:
        """
        Create output directories and return paths
        """
        paths = {}
        for name, dirname in cls.OUTPUT_DIRS.items():
            path = cls.PROJECT_ROOT / dirname
            path.mkdir(exist_ok=True)
            paths[name] = path
        return paths
    
    @classmethod
    def save_config_to_file(cls, config: Dict[str, Any], filepath: Path):
        """
        Save configuration to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    @classmethod
    def load_config_from_file(cls, filepath: Path) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], config_type: str = 'env') -> bool:
        """
        Validate configuration dictionary
        """
        if config_type == 'env':
            required_keys = ['gb_path', 'init_state', 'max_steps']
        elif config_type == 'epsilon':
            required_keys = ['epsilon_start', 'epsilon_min', 'epsilon_decay']
        elif config_type == 'comparison':
            required_keys = ['num_episodes', 'max_steps_per_episode']
        else:
            return True
        
        return all(key in config for key in required_keys)
    
    @classmethod
    def get_testing_config(cls) -> Dict[str, Any]:
        """
        Get configuration optimized for testing (faster execution)
        """
        return {
            'env': cls.get_env_config(
                max_steps=10000,
                headless=True,
                save_video=False
            ),
            'comparison': cls.get_comparison_config(
                num_episodes=2,
                max_steps_per_episode=10000,
                create_visualizations=False
            ),
            'epsilon': cls.get_epsilon_config(
                epsilon_start=0.8,
                epsilon_decay=0.99
            )
        }


# Global configuration instance
config = Config()


# Utility functions for common configuration tasks
def get_session_config(session_name: str, **overrides) -> Dict[str, Any]:
    """
    Get configuration for a specific session
    """
    session_path = config.PROJECT_ROOT / "sessions" / session_name
    session_path.mkdir(parents=True, exist_ok=True)
    
    env_config = config.get_env_config(session_path=session_path, **overrides)
    return env_config


def get_experiment_config(experiment_name: str, 
                         num_episodes: int = 5,
                         **overrides) -> Dict[str, Dict[str, Any]]:
    """
    Get complete configuration for an experiment
    """
    experiment_dir = config.PROJECT_ROOT / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'env': get_session_config(f"exp_{experiment_name}", **overrides),
        'comparison': config.get_comparison_config(num_episodes=num_episodes),
        'epsilon': config.get_epsilon_config(),
        'experiment_dir': experiment_dir
    }


def setup_experiment_environment(experiment_name: str) -> Dict[str, Path]:
    """
    Setup directories and configuration for an experiment
    """
    experiment_dir = config.PROJECT_ROOT / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    subdirs = {}
    for subdir in ['results', 'logs', 'visualizations', 'configs']:
        path = experiment_dir / subdir
        path.mkdir(exist_ok=True)
        subdirs[subdir] = path
    
    return subdirs
