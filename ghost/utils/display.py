"""
ghost/utils/display.py
All CLI visual candy: ASCII art, progress bars, styled prints.
"""
import subprocess

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

GHOST_LOGO = f"""{CYAN}{BOLD}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⣤⡤⣤⢤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠛⠉⠀⠀⠀⠀⠀⠀⠉⠻⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠏⠀⠀⠀⠀⣀⡀⠀⠀⠀⠀⠀⠈⢻⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠯⠀⠀⠀⢰⣿⣿⣿⣆⡀⣴⣾⣿⣦⡈⢿⡄⠀  G H O S T
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣏⠀⠀⠀⠀⢸⣿⣿⣿⡿⠇⣿⣿⣿⣿⠆⠘⣷⡀  Generalizable Hyperspectral
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⠀⠀⠈⠿⠿⡿⠃⠀⢿⣿⣿⠏⠀⠀⢹⣗  Observation & Segmentation
⠀⠀⠀⠀⠀⠀⠀⠀⢸⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⣿⡀ Toolkit
{RESET}"""

# Blushing ghost — used by predict.py
GHOST_PREDICT = f"""{CYAN}{BOLD}
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠟⠋⠙⠛⠉⠙⠻⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⣤⡤⢄⣀⣠⡏⠁⠀⠀⠀⠀⠀⠀⠀⠀⢳⣀⣠⣤⣤⣄⠀⠀⠀
⠀⠀⠀⣸⣿⣦⡀⠀⠉⠙⠲⠤⠤⠤⠤⠤⠤⠴⠚⠉⠁⠀⣠⣾⣿⡀⠀⠀
⠀⣀⣀⣿⣿⣿⣿⡶⢤⣀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠴⣾⣿⣿⣿⣇⣀⠀
⣾⠋⠙⣻⣿⣿⡟⠀⠀⠈⣭⣟⠒⠒⠒⠒⢲⣯⡉⠀⠀⠈⢿⣿⡟⠁⠙⣧
⣿⠀⠀⠘⢿⡟⠀⠀⠀⠸⣿⣿⠀⠀⠀⠀⣿⣿⡇⠀⠀⠀⠘⡿⠃⠀⠀⢻
⢻⣆⠀⠀⠀⠀⠀⠀⠀⡀⠉⠁⠀⠀⠀⠀⠈⠉⠀⠀⠀⠀⠀⠀⠀⠀⢠⡾
⠀⠹⣤⡀⠀⠀⠀⠀⠀⢿⣶⣤⣤⣤⣤⣤⣤⣶⡾⠀⠀⠀⠀⠀⢀⣠⠞⠀
⠀⠀⠈⠛⣶⠀⠀⠀⠀⠀⠙⠿⢟⣉⣉⣻⠿⠋⠀⠀⠀⠀⢠⣶⠟⠁⠀⠀
⠀⠀⠀⠀⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡏⠀⠀⠀⠀
⠀⠀⠀⠀⠸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢻⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢾⠇⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠻⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠷⢦⣄⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⢳⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢶⣠⣤⣤⣤⣤⣤⣤⣤⣤⣴⠾⠛⠁⠀⠀⠀
⠀  Predicting...
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
⠀  Visualizing...
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
    Single-line progress bar that fills 0→100% across FULL training.

    Between validation boundaries: shows epoch/total + bar + loss.
    At validation boundaries: same line but with metrics, printed as newline.
    Uses \\r to overwrite between boundaries.
    """
    pct = epoch / total
    filled = int(BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    pct_int = int(100 * pct)

    bar_col = GREEN if pct >= 0.5 else YELLOW
    bar_str = _c(bar, bar_col)

    parts = [
        f"{prefix}{BOLD}Epoch {epoch:3d}/{total}{RESET}",
        f"{bar_str} {_c(f'{pct_int:3d}%', GRAY)}",
        f"Loss {_c(f'{loss:.4f}', CYAN)}",
    ]

    at_boundary = (epoch % interval == 0) or (epoch == total)

    if at_boundary and oa is not None:
        col = GREEN if oa >= 0.9 else YELLOW if oa >= 0.7 else RED
        parts.append(f"OA {_c(f'{oa:.4f}', col)}")
    if at_boundary and miou is not None:
        col = GREEN if miou >= 0.9 else YELLOW if miou >= 0.7 else RED
        parts.append(f"mIoU {_c(f'{miou:.4f}', col)}")
    if at_boundary and kappa is not None:
        col = GREEN if kappa >= 0.9 else YELLOW if kappa >= 0.7 else RED
        parts.append(f"κ {_c(f'{kappa:.4f}', col)}")

    line = " | ".join(parts)

    if at_boundary:
        print(f"\r{line}")
    else:
        print(f"\r{line}", end="", flush=True)


def forest_banner(forest_idx: int, num_forests: int, seed: int,
                  node_id: str) -> str:
    return (
        f"\n  {BOLD}{CYAN}── Ensemble {forest_idx+1}/{num_forests}{RESET}"
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
        f"Ensembles: {num_forests}   Loss: {loss_str}",
        f"  Train pixels  : {train_pixels}   Val pixels: {val_pixels}",
        f"{BOLD}{MAGENTA}{'═'*60}{RESET}",
    ]
    return "\n".join(lines)


def gpu_stats() -> str:
    """Get real GPU stats (temp, utilization, VRAM) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip().split(',')
        temp, util, used, total = [s.strip() for s in out]
        return f"GPU {temp}°C | {used} / {total} MB | {util}% util"
    except Exception:
        return ""


def forest_done_line(forest_idx: int, num_forests: int,
                     best_miou: float, best_epoch: int,
                     best_oa: float, best_aa: float, best_kappa: float,
                     elapsed: str) -> str:
    miou_col = GREEN if best_miou >= 0.65 else YELLOW if best_miou >= 0.45 else RED
    oa_col   = GREEN if best_oa >= 0.9 else YELLOW if best_oa >= 0.7 else RED
    aa_col   = GREEN if best_aa >= 0.9 else YELLOW if best_aa >= 0.7 else RED
    k_col    = GREEN if best_kappa >= 0.85 else YELLOW if best_kappa >= 0.65 else RED
    gpu = gpu_stats()
    hw_line = f"    {GRAY}{elapsed} elapsed{f' | {gpu}' if gpu else ''}{RESET}"
    return (
        f"  {GREEN}✓{RESET} Ensemble {forest_idx+1}/{num_forests} done"
        f"  Best @ epoch {BOLD}{best_epoch}{RESET}"
        f"  mIoU {_c(f'{best_miou:.4f}', miou_col)}"
        f"  OA {_c(f'{best_oa:.4f}', oa_col)}"
        f"  AA {_c(f'{best_aa:.4f}', aa_col)}"
        f"  κ {_c(f'{best_kappa:.4f}', k_col)}"
        f"\n{hw_line}"
    )


def print_logo():
    print(GHOST_LOGO)


def print_predict_start():
    print(GHOST_PREDICT)


def print_visualize_start():
    print(GHOST_FLOWER)


def print_training_start():
    print(GHOST_TRAINING)


def print_training_done():
    print(GHOST_DONE)


def print_config_box(title: str, items: list[tuple[str, str]]):
    """
    Print a styled ═══ config box.
    items: list of (label, value) tuples.
    """
    W = 60
    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"  {BOLD}{CYAN}{title}{RESET}")
    for label, value in items:
        print(f"  {label:<14}: {value}")
    print(f"{BOLD}{'═' * W}{RESET}")


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
    print(f"\n{GRAY}  # Evaluate all routing modes:{RESET}")
    print(f"  ghost predict \\")
    print(f"    --data  {data_path} \\")
    print(f"    --gt    {gt_path} \\")
    print(f"    --model {model_path} \\")
    print(f"    --routing all --out-dir {out_dir}")
    print(f"\n{GRAY}  # Visualize predictions:{RESET}")
    print(f"  ghost visualize \\")
    print(f"    --data  {data_path} \\")
    print(f"    --gt    {gt_path} \\")
    print(f"    --model {model_path} \\")
    print(f"    --out-dir {out_dir}\n")