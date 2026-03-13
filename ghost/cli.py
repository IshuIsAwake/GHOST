import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="GHOST: Generalizable Hyperspectral Observation Segmentation Tool",
        usage="ghost <command> [<args>]"
    )
    
    parser.add_argument('command', help='Subcommand to run: train, train_rssp, predict')
    
    # parse_known_args allows the main CLI to ignore arguments that are meant for subcommands
    args, _ = parser.parse_known_args(sys.argv[1:2])
    
    if args.command == 'train':
        from ghost.train import main as run_train
        # Reset sys.argv so argparse inside train.py doesn't see 'ghost train'
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run_train()
    
    elif args.command == 'train_rssp':
        from ghost.train_rssp import main as run_train_rssp
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run_train_rssp()
        
    elif args.command == 'predict':
        from ghost.predict import main as run_predict
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run_predict()
        
    else:
        print("Unrecognized command")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
