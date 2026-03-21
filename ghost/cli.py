import sys
import argparse
from ghost.utils.display import (
    print_logo, GHOST_BOO, GHOST_FLOWER, RESET, CYAN, BOLD, GRAY
)


def main():
    # Logo on every invocation
    print_logo()

    parser = argparse.ArgumentParser(
        description="GHOST: Generalizable Hyperspectral Observation & Segmentation Toolkit",
        usage="ghost <command> [<args>]"
    )
    from ghost import __version__
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'ghost-hsi {__version__}'
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

    elif args.command == 'demo':
        from ghost.data import indian_pines_path
        data_path, gt_path = indian_pines_path()
        print(f"{BOLD}Bundled Indian Pines dataset:{RESET}")
        print(f"  Data : {data_path}")
        print(f"  GT   : {gt_path}")
        print(f"\n{BOLD}{CYAN}Run this to start training:{RESET}")
        print(f"\n  ghost train_rssp \\")
        print(f"    --data  {data_path} \\")
        print(f"    --gt    {gt_path} \\")
        print(f"    --loss dice --routing forest \\")
        print(f"    --base_filters 32 --num_filters 8 \\")
        print(f"    --forests 5 --leaf_forests 3 \\")
        print(f"    --epochs 400 --patience 50 --min_epochs 40 \\")
        print(f"    --out-dir runs/indian_pines")

    elif args.command == 'version':
        from ghost import __version__
        print(f"ghost-hsi v{__version__}")

    else:
        print(f"{BOLD}Unrecognized command: '{args.command}'{RESET}")
        print(f"{GRAY}Available commands: train | train_rssp | predict | visualize | demo | version | boo | flower")
        print(f"Use 'ghost --version' or 'ghost version' to check installed version{RESET}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
