"""
Environment Setup and Verification Script
=========================================

This script helps set up and verify the Pokemon Red comparison environment.
"""

import sys
import subprocess
import os
from pathlib import Path
import importlib.util


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ Error: Python 3 is required")
        return False
    
    if version.minor < 10:
        print("⚠️  Warning: Python 3.10+ is recommended")
        return True
    elif version.minor > 11:
        print("⚠️  Warning: Python versions > 3.11 may have CUDA compatibility issues")
        return True
    else:
        print("✅ Python version is compatible")
        return True


def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"❌ {package_name} not found")
            return False
        
        # Try to import
        module = importlib.import_module(import_name)
        
        # Get version if available
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
        
    except ImportError as e:
        print(f"❌ {package_name} import error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name} warning: {e}")
        return True


def check_required_packages():
    """Check all required packages"""
    print("\n" + "="*50)
    print("CHECKING REQUIRED PACKAGES")
    print("="*50)
    
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('pytorch', 'torch'),
        ('stable-baselines3', 'stable_baselines3'),
        ('pyboy', 'pyboy'),
        ('gymnasium', 'gymnasium'),
        ('mediapy', 'mediapy'),
        ('einops', 'einops'),
        ('pillow', 'PIL'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('tqdm', 'tqdm'),
        ('yaml', 'yaml'),
    ]
    
    success_count = 0
    total_count = len(required_packages)
    
    for package_name, import_name in required_packages:
        if check_package(package_name, import_name):
            success_count += 1
    
    print(f"\nPackage check: {success_count}/{total_count} packages available")
    return success_count == total_count


def check_cuda_support():
    """Check CUDA support"""
    print("\n" + "="*50)
    print("CHECKING CUDA SUPPORT")
    print("="*50)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"✅ CUDA available: {cuda_available}")
            print(f"✅ GPU devices: {device_count}")
            print(f"✅ Current device: {device_name}")
            
            # Test GPU memory
            try:
                torch.cuda.empty_cache()
                total_memory = torch.cuda.get_device_properties(current_device).total_memory
                print(f"✅ GPU memory: {total_memory / 1024**3:.1f} GB")
            except:
                print("⚠️  Could not check GPU memory")
                
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("   This is fine but training will be slower")
            
        return True
        
    except ImportError:
        print("❌ PyTorch not installed - cannot check CUDA")
        return False
    except Exception as e:
        print(f"⚠️  CUDA check warning: {e}")
        return True


def check_game_files():
    """Check if Pokemon Red game files are available"""
    print("\n" + "="*50)
    print("CHECKING GAME FILES")
    print("="*50)
    
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    # Check for ROM file
    rom_paths = [
        parent_dir / "PokemonRed.gb",
        current_dir / "PokemonRed.gb",
        parent_dir / "v2" / "PokemonRed.gb"
    ]
    
    rom_found = False
    for rom_path in rom_paths:
        if rom_path.exists():
            print(f"✅ ROM file found: {rom_path}")
            rom_found = True
            break
    
    if not rom_found:
        print("❌ PokemonRed.gb not found in expected locations")
        print("   Please ensure the ROM file is available")
    
    # Check for init state
    state_paths = [
        parent_dir / "init.state",
        current_dir / "init.state",
        parent_dir / "v2" / "init.state"
    ]
    
    state_found = False
    for state_path in state_paths:
        if state_path.exists():
            print(f"✅ Init state found: {state_path}")
            state_found = True
            break
    
    if not state_found:
        print("❌ init.state not found in expected locations")
        print("   Please ensure the initial state file is available")
    
    return rom_found and state_found


def check_output_directories():
    """Check and create output directories"""
    print("\n" + "="*50)
    print("CHECKING OUTPUT DIRECTORIES")
    print("="*50)
    
    current_dir = Path(__file__).parent
    
    directories = [
        'comparison_results',
        'metrics_analysis', 
        'experiments',
        'logs',
        'sessions'
    ]
    
    for dirname in directories:
        dir_path = current_dir / dirname
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")
    
    return True


def run_basic_test():
    """Run a basic functionality test"""
    print("\n" + "="*50)
    print("RUNNING BASIC FUNCTIONALITY TEST")
    print("="*50)
    
    try:
        # Test epsilon greedy agent import
        from epsilon_greedy_agent import EpsilonGreedyAgent
        print("✅ Epsilon Greedy agent import successful")
        
        # Create test agent
        agent = EpsilonGreedyAgent(
            epsilon_start=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.99
        )
        print("✅ Agent creation successful")
        
        # Test action selection with dummy observation
        import numpy as np
        dummy_obs = {
            'screens': np.zeros((72, 80, 3), dtype=np.uint8),
            'health': np.array([1.0]),
            'level': np.zeros(8),
            'badges': np.zeros(8, dtype=np.int8),
            'events': np.zeros(100, dtype=np.int8),
            'map': np.zeros((48, 48, 1), dtype=np.uint8),
            'recent_actions': np.zeros(3, dtype=np.uint8)
        }
        
        action = agent.select_action(dummy_obs)
        print(f"✅ Action selection successful: action={action}")
        
        # Test metrics
        metrics = agent.get_performance_metrics()
        print(f"✅ Metrics collection successful: {len(metrics)} metrics")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        print("   Make sure you're in the comparison_agents directory")
        return False
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def main():
    """Main verification function"""
    print("Pokemon Red Agent Comparison - Environment Verification")
    print("=" * 80)
    
    all_checks = []
    
    # Run all checks
    all_checks.append(check_python_version())
    all_checks.append(check_required_packages())
    all_checks.append(check_cuda_support())
    all_checks.append(check_game_files())
    all_checks.append(check_output_directories())
    all_checks.append(run_basic_test())
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed_checks = sum(all_checks)
    total_checks = len(all_checks)
    
    if passed_checks == total_checks:
        print(f"✅ All checks passed ({passed_checks}/{total_checks})")
        print("🚀 Environment is ready for Pokemon Red agent comparison!")
        print("\nNext steps:")
        print("1. python run_comparison.py --mode standalone --episodes 2")
        print("2. python example_usage.py")
    else:
        print(f"⚠️  Some checks failed ({passed_checks}/{total_checks})")
        print("❌ Please fix the issues above before proceeding")
        
        if not any([all_checks[1], all_checks[5]]):  # packages or basic test
            print("\nQuick fix suggestions:")
            print("1. conda activate pokemon-red-comparison")
            print("2. pip install -r requirements_py310.txt")
            print("3. python verify_environment.py")
    
    return passed_checks == total_checks


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)