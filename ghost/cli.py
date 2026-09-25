import os
import sys
import argparse
from ghost.utils.display import (
    print_logo, GHOST_FLOWER, RESET, CYAN, BOLD, GRAY, RED
)

ARCH_COMMANDS = ('train', 'train_spt', 'train_rssp', 'predict', 'visualize')


def _pop_arch(argv):
    """Remove --arch X / --arch=X so the architecture's own parser never sees it."""
    arch, rest, i = None, [], 0
    while i < len(argv):
        if argv[i] == '--arch':
            if i + 1 >= len(argv):
                raise ValueError("--arch needs a version, e.g. --arch 0.2.0")
            arch, i = argv[i + 1], i + 2
        elif argv[i].startswith('--arch='):
            arch, i = argv[i].split('=', 1)[1], i + 1
        else:
            rest.append(argv[i])
            i += 1
    return arch, rest


def _model_path(argv):
    for i, a in enumerate(argv):
        if a == '--model' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--model='):
            return a.split('=', 1)[1]
    return None


def _resolve_entry(command, arch_flag, rest):
    """Pick the architecture: train uses --arch (default latest), train_spt is 0.1.7-only,
    predict/visualize follow the checkpoint and treat --arch as an assertion."""
    from ghost.archs import detect_checkpoint_arch, get_entry, resolve_arch
    if command in ('train_spt', 'train_rssp'):
        return get_entry(resolve_arch(arch_flag) if arch_flag else '0.1.7', 'train_spt')
    if command in ('predict', 'visualize'):
        model = _model_path(rest)
        if model and os.path.isfile(model):
            arch = detect_checkpoint_arch(model)
            if arch_flag and resolve_arch(arch_flag) != arch:
                raise ValueError(f"{model} was trained with architecture {arch}, not {resolve_arch(arch_flag)}")
            return get_entry(arch, command)
    return get_entry(resolve_arch(arch_flag), command)


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
        help='Subcommand: train | train_spt | predict | visualize | convert_to_mat | demo | version | flower'
    )

    args, _ = parser.parse_known_args(sys.argv[1:2])

    if args.command in ARCH_COMMANDS:
        try:
            arch_flag, rest = _pop_arch(sys.argv[2:])
            entry = _resolve_entry(args.command, arch_flag, rest)
        except ValueError as exc:
            print(f"{RED}{BOLD}  {exc}{RESET}")
            sys.exit(2)
        sys.argv = [sys.argv[0]] + rest
        entry()

    elif args.command == 'convert_to_mat':
        from ghost.convert import main as run_convert
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        run_convert()

    elif args.command == 'flower':
        print(f"{CYAN}{GHOST_FLOWER}{RESET}")

    elif args.command == 'demo':
        from ghost.data import indian_pines_path
        data_path, gt_path = indian_pines_path()
        print(f"{BOLD}Bundled Indian Pines dataset:{RESET}")
        print(f"  Data : {data_path}")
        print(f"  GT   : {gt_path}")
        print(f"\n{BOLD}{CYAN}Run this to start training (architecture 0.2.0, per-pixel):{RESET}")
        print(f"\n  ghost train \\")
        print(f"    --data  {data_path} \\")
        print(f"    --gt    {gt_path} \\")
        print(f"    --loss dice \\")
        print(f"    --out-dir runs/indian_pines_v0_2")
        print(f"\n{GRAY}The v0.1.7 pipeline (3-D conv + U-Net with SPT) is still available:{RESET}")
        print(f"\n  ghost train_spt \\")
        print(f"    --data  {data_path} \\")
        print(f"    --gt    {gt_path} \\")
        print(f"    --loss dice \\")
        print(f"    --base_filters 32 --num_filters 8 \\")
        print(f"    --ensembles 5 --leaf_ensembles 3 \\")
        print(f"    --epochs 400 --patience 50 --min_epochs 40 \\")
        print(f"    --out-dir runs/indian_pines")

    elif args.command == 'version':
        from ghost import __version__
        from ghost.archs import ARCHS, DEFAULT_ARCH
        print(f"ghost-hsi v{__version__}")
        print("architectures: " + ", ".join(a + (" (default)" if a == DEFAULT_ARCH else "") for a in ARCHS))

    else:
        print(f"{BOLD}Unrecognized command: '{args.command}'{RESET}")
        print(f"{GRAY}Available commands: train | train_spt | predict | visualize | convert_to_mat | demo | version | flower")
        print(f"Use 'ghost --version' or 'ghost version' to check installed version{RESET}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
