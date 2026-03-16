import sys
import argparse
from ghost.utils.display import (
    print_logo, GHOST_BOO, GHOST_FLOWER, RESET, CYAN, BOLD, GRAY
)


def main():
    # Logo on every invocation
    print_logo()

    parser = argparse.ArgumentParser(
        description="GHOST: Generalizable Hyperspectral Observation Segmentation Tool",
        usage="ghost <command> [<args>]"
    )
    parser.add_argument(
        'command',
        help='Subcommand: train | train_rssp | predict | visualize | boo | flower'
    )

    args, _ = parser.parse_known_args(sys.argv[1:2])

    if args.command == 'train':
        from ghost.train import main as run_train
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

    elif args.command == 'visualize':
        from ghost.visualize import main as run_visualize
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run_visualize()

    elif args.command == 'boo':
        print(GHOST_BOO)

    elif args.command == 'flower':
        print(f"{CYAN}{GHOST_FLOWER}{RESET}")

    else:
        print(f"{BOLD}Unrecognized command: '{args.command}'{RESET}")
        print(f"{GRAY}Available commands: train | train_rssp | predict | visualize | boo | flower{RESET}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()