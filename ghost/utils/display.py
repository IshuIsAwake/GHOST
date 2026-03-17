"""
ghost/utils/display.py
All CLI visual candy: ASCII art, progress bars, styled prints.
"""

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"

def _c(text, *codes):
    return "".join(codes) + str(text) + RESET


# ── ASCII art ─────────────────────────────────────────────────────────────────

# Small ghost shown on every ghost command invocation
GHOST_LOGO = f"""{CYAN}{BOLD}
  .--.
 (o  o)  G H O S T
 | O  |  Generalizable Hyperspectral
  \\--/   Observation & Segmentation Toolkit
  ~~~~
{RESET}"""

# Three-ghost boo screen
GHOST_BOO = f"""{CYAN}{BOLD}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠈⠈⠈⠈⠀⠀⠛⠀⠀⠙⠃⠀⠈⠋⠄⠀⠙⠁⠀⠘⠃⠀⠀⚄⠀⠀⠀⠀⠀⠀⠂⠀⠀⠙⠁⠀⠈⠂⠀⠀⠛⠀⠀⠈⠀⠀⠀⠂⠀⠀⠙⠁⠀⠈⠊⠀⠀⠛⠀⠀⠘⠃⠀⠀⠋
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠀⠈⠀⠀⠠⠀⠀⠀⡄⠀⠀⠠⠀⠀⠀⠄⠀⠀⠄⠀⠀⠠⠀⠀⠀⡄⠀⠀⢠⠀⠀⠀⠄⠀⠀⠄⠀⠀⠠⠀⢀⣠⣶⣼⣿⣿⣿⣿⣿⣿⣶⣤⣤⠀⠀⠠⠀⠀⠀
      B  O  O  !
{RESET}"""

GHOST_FLOWER = f"""{CYAN}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠤⠒⠒⠒⠒⠢⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⠀⠀⠀⠀⠀⡤⢄⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⡰⠃⠀⠀⠀⠀⠀⠀⠀⣼⡆⠀⢀⢀⠀⠸⡟⡇⠀⠀⡰⢾⢁⢸⣄⠀
⠀⠀⠀⠀⠀⠀⠀⣰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠙⠈⠀⠈⠉⠀⠀⠀⢸⠀⠀⡝⡆⠚⠠⣤⠇
⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡀⠀⠙⢸⠧⣤⠃⠀
⠀⠀⠀⠀⠀⢠⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣇⡀⣤⣼⠧⡀⠀⠀
⠀⠀⠀⠀⢠⡏⠀⠀⠀⠀⡀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⡇⠀⣀⣧⢸⠀⠀⠀
⠀⠀⠀⢠⡞⠀⠀⠀⠀⠀⢟⠀⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠒⠉⠁⢸⡄⠀⠀⠀
⠀⠀⢀⡞⠀⠀⠀⠀⠀⠀⢸⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠈⠀⠀⠀⠀
⠀⣠⠏⠀⠀⠀⠀⠀⠀⠀⠈⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⠀⠀⠀⠀⠀⠀
⣰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡏⠀⠀⠀⠀⠀⠀⠀⠀
⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠙⠶⠦⠤⠶⠖⠒⢤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡴⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⣷⠀⠀⠀⠀⢀⣤⠴⠶⣤⣀⠀⠀⣠⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠙⠲⠤⠖⠚⠉⠀⠀⠀⠀⠈⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
{RESET}"""

# Ghost shown when any training command starts
GHOST_TRAINING = f"""{CYAN}{BOLD}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⣤⡤⣤⢤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠛⠉⠀⠀⠀⠀⠀⠀⠉⠻⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠏⠀⠀⠀⠀⣀⡀⠀⠀⠀⠀⠀⠈⢻⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠯⠀⠀⠀⢰⣿⣿⣿⣆⡀⣴⣾⣿⣦⡈⢿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣏⠀⠀⠀⠀⢸⣿⣿⣿⡿⠇⣿⣿⣿⣿⠆⠘⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⠀⠀⠈⠿⠿⡿⠃⠀⢿⣿⣿⠏⠀⠀⢹⣗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢸⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀  Training initiated...
{RESET}"""

# Ghost shown when training completes
GHOST_DONE = f"""{GREEN}{BOLD}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⡶⠟⠛⠛⠛⠉⠉⠛⠛⠻⢶⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣤⠀⠀⠀⠀⠀⣠⣤⡀⢀⣀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣷⡀⠀⠀⢸⡏⠀⠹⠋⢹⣧
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⡀⠀⠀⠻⢶⣤⣴⠾⠃
⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠃⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣧⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣾⡇⠀⠀⠀⢿⣿⡏⠀⡀⠀⡀⠀⣾⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣿⠁⠐⡀⢠⠀⠀⡄⠀⠙⠛⠁⠀⠈⠉⠁⠀⠐⠄⠀⠀⠀⠀⠀⠀⠀⢿⣇⠀⠀⠀⠀⠀⠀⠀
⠀  Training complete!
{RESET}"""


# ── Progress bar ──────────────────────────────────────────────────────────────

BAR_WIDTH = 24  # characters

def epoch_bar(epoch: int, total: int,
              loss: float,
              val_loss: float = None,
              oa: float = None,
              miou: float = None,
              aa: float = None,
              kappa: float = None,
              interval: int = 20,
              prefix: str = "  ") -> None:
    """
    Print an animated progress bar that fills 0→100% within each interval,
    then locks in place (newline) when the interval is complete.

    Call this every epoch. It uses \\r to overwrite the current line while
    filling within the interval, then prints a newline at interval boundaries.

    interval: epochs between each locked checkpoint line (default 20)
    """
    # Bar and percentage based on overall progress (0 → 100% across full training)
    overall_pct = epoch / total
    filled = int(BAR_WIDTH * overall_pct)
    bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
    pct    = int(100 * overall_pct)

    bar_col = GREEN if overall_pct >= 0.5 else YELLOW
    bar_str = _c(bar, bar_col)

    parts = [
        f"{prefix}{BOLD}Epoch {epoch:4d}/{total}{RESET}",
        f"{bar_str} {_c(f'{pct:3d}%', GRAY)}",
        f"Loss {_c(f'{loss:.4f}', CYAN)}",
    ]

    if val_loss is not None:
        parts.append(f"ValLoss {_c(f'{val_loss:.4f}', BLUE)}")
    if oa is not None:
        col = GREEN if oa >= 0.9 else YELLOW if oa >= 0.7 else RED
        parts.append(f"OA {_c(f'{oa:.4f}', col)}")
    if miou is not None:
        col = GREEN if miou >= 0.9 else YELLOW if miou >= 0.7 else RED
        parts.append(f"mIoU {_c(f'{miou:.4f}', col)}")
    if aa is not None:
        col = GREEN if aa >= 0.9 else YELLOW if aa >= 0.7 else RED
        parts.append(f"AA {_c(f'{aa:.4f}', col)}")
    if kappa is not None:
        col = GREEN if kappa >= 0.9 else YELLOW if kappa >= 0.7 else RED
        parts.append(f"κ {_c(f'{kappa:.4f}', col)}")

    line = " | ".join(parts)

    at_boundary = (epoch % interval == 0) or (epoch == total)
    if at_boundary:
        # Lock this line — move to next line
        print(f"\r{line}")
    else:
        # Overwrite current line — no newline
        print(f"\r{line}", end="", flush=True)


def forest_banner(forest_idx: int, num_forests: int, seed: int,
                  node_id: str) -> str:
    return (
        f"\n  {BOLD}{CYAN}── Forest {forest_idx+1}/{num_forests}{RESET}"
        f"  seed={seed}  node='{node_id}'"
    )


def node_banner(node_id: str, node_classes: list, num_classes: int,
                epochs: int, num_forests: int, loss_type: str,
                focal_gamma: float, train_pixels: int,
                val_pixels: int) -> str:
    loss_str = loss_type + (f" γ={focal_gamma}" if "focal" in loss_type else "")
    lines = [
        "",
        f"{BOLD}{MAGENTA}{'═'*60}{RESET}",
        f"  {BOLD}Node{RESET} {CYAN}'{node_id}'{RESET}",
        f"  Classes       : {node_classes}",
        f"  Local classes : {num_classes - 1}   Epochs: {epochs}   "
        f"Forests: {num_forests}   Loss: {loss_str}",
        f"  Train pixels  : {train_pixels}   Val pixels: {val_pixels}",
        f"{BOLD}{MAGENTA}{'═'*60}{RESET}",
    ]
    return "\n".join(lines)


def forest_done_line(forest_idx: int, num_forests: int,
                     best_miou: float, node_elapsed: str,
                     global_elapsed: str, vram: str) -> str:
    miou_col = GREEN if best_miou >= 0.65 else YELLOW if best_miou >= 0.45 else RED
    return (
        f"  {GREEN}✓{RESET} Forest {forest_idx+1}/{num_forests} done"
        f"  Best mIoU {_c(f'{best_miou:.4f}', miou_col)}"
        f"  {GRAY}node {node_elapsed}  total {global_elapsed}  {vram}{RESET}"
    )


def print_logo():
    print(GHOST_LOGO)


def print_training_start():
    print(GHOST_TRAINING)


def print_training_done():
    print(GHOST_DONE)


def print_results_box(metrics: dict, routing: str = None):
    """
    Print a styled results box.
    metrics: dict with keys like 'OA', 'mIoU', 'Dice', 'Precision', 'Recall', 'AA', 'kappa'
    """
    title = "Test Results" + (f" [{routing}]" if routing else "")
    w = 42
    print(f"\n{BOLD}{GREEN}{'─'*w}{RESET}")
    print(f"{BOLD}{GREEN}  {title}{RESET}")
    print(f"{BOLD}{GREEN}{'─'*w}{RESET}")

    colour_rules = {
        'OA':        lambda v: GREEN if v >= 0.90 else YELLOW if v >= 0.75 else RED,
        'mIoU':      lambda v: GREEN if v >= 0.65 else YELLOW if v >= 0.45 else RED,
        'AA':        lambda v: GREEN if v >= 0.65 else YELLOW if v >= 0.45 else RED,
        'kappa':     lambda v: GREEN if v >= 0.85 else YELLOW if v >= 0.65 else RED,
        'Dice':      lambda v: GREEN if v >= 0.70 else YELLOW,
        'Precision': lambda v: GREEN if v >= 0.70 else YELLOW,
        'Recall':    lambda v: GREEN if v >= 0.70 else YELLOW,
    }

    for key, val in metrics.items():
        col = colour_rules.get(key, lambda v: WHITE)(val)
        bar_len = int(val * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {key:<12} {_c(f'{val:.4f}', col, BOLD)}  {_c(bar, col)}")

    print(f"{BOLD}{GREEN}{'─'*w}{RESET}\n")


def print_per_class_iou(class_ious: dict, pixel_counts: dict = None):
    """
    Print per-class IoU with colour coding.
    If pixel_counts is provided (dict {class_id: int}), show pixel counts and
    flag classes with fewer than 20 pixels.

    IoU colour thresholds: green ≥ 0.8, yellow ≥ 0.5, red < 0.5
    """
    print(f"\n{BOLD}  Per-class IoU:{RESET}")
    for c, iou in class_ious.items():
        col = GREEN if iou >= 0.8 else YELLOW if iou >= 0.5 else RED
        bar = "█" * int(iou * 20)
        if pixel_counts is not None:
            px      = pixel_counts.get(c, 0)
            warning = f"  {YELLOW}⚠ few pixels{RESET}" if px < 20 else ""
            print(f"  Class {c:2d}  {_c(f'{iou:.4f}', col)}  {_c(bar, col):<20s}"
                  f"  {GRAY}({px:>5d} px){RESET}{warning}")
        else:
            print(f"  Class {c:2d}  {_c(f'{iou:.4f}', col)}  {_c(bar, col)}")
    print()


def print_save_and_next(out_dir: str, save_file: str,
                        data_path: str, gt_path: str,
                        train_ratio: float, val_ratio: float):
    """Print save confirmation and suggest next commands."""
    model_path = f"{out_dir}/{save_file}"
    print(f"\n{GREEN}{BOLD}  Saved →{RESET} {model_path}")
    print(f"\n{BOLD}{CYAN}  What's next?{RESET}")
    print(f"{GRAY}  ┌─ Evaluate all routing modes:{RESET}")
    print(f"  │  ghost predict \\")
    print(f"  │    --data  {data_path} \\")
    print(f"  │    --gt    {gt_path} \\")
    print(f"  │    --model {model_path} \\")
    print(f"  │    --routing all --out-dir {out_dir}")
    print(f"{GRAY}  │{RESET}")
    print(f"{GRAY}  └─ Visualize predictions:{RESET}")
    print(f"     ghost visualize \\")
    print(f"       --data  {data_path} \\")
    print(f"       --gt    {gt_path} \\")
    print(f"       --model {model_path} \\")
    print(f"       --out-dir {out_dir}\n")