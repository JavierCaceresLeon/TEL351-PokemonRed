import argparse
from train_core import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid Sage Agent")
    parser.add_argument("--scenario", required=True, help="Scenario ID (e.g., pewter_brock)")
    parser.add_argument("--phase", default="battle", help="Phase name (default: battle)")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--no-headless", action="store_true", help="Show GameBoy window")
    parser.add_argument("--run-name", type=str, default="")
    
    args = parser.parse_args()
    
    run_training("hybrid", args.scenario, args.phase, args.timesteps, not args.no_headless, args.run_name)
