import numpy as np

def compute_class_means(data, labels, num_classes):
    """
    Compute mean spectrum for each class.
    data:   (C, H, W) tensor or numpy array
    labels: (H, W) numpy array
    Returns: (num_classes, C) numpy array
    """
    if hasattr(data, 'numpy'):
        data = data.numpy()
    
    C, H, W = data.shape
    data_hw = data.reshape(C, -1).T  # (H*W, C)
    labels_flat = labels.reshape(-1) # (H*W,)
    
    means = np.zeros((num_classes, C), dtype=np.float32)
    for c in range(1, num_classes):
        mask = labels_flat == c
        if mask.sum() > 0:
            means[c] = data_hw[mask].mean(axis=0)
    
    return means  # (num_classes, C)


def continuum_removal_numpy(spectra):
    """
    Apply continuum removal to a batch of spectra.
    spectra: (N, C) numpy array
    Returns: (N, C) numpy array
    """
    N, C = spectra.shape
    band_indices = np.linspace(0, 1, C)
    
    out = np.zeros_like(spectra)
    for i in range(N):
        s = spectra[i]
        s_min = s.min()
        s_max = s.max()
        continuum = s_min + (s_max - s_min) * band_indices
        out[i] = s / (continuum + 1e-8)
    
    return out


def sam_distance(a, b):
    """
    Spectral Angle Mapper distance between two spectra.
    Returns angle in radians (0 = identical, π/2 = orthogonal)
    """
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    cos_angle = np.clip(dot / (norm + 1e-8), -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_sam_matrix(means):
    """
    Compute pairwise SAM distance matrix.
    means: (num_classes, C)
    Returns: (num_classes, num_classes) distance matrix
    """
    N = len(means)
    matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            d = sam_distance(means[i], means[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def spectral_balanced_split(classes, pixel_counts, sam_matrix):
    """
    Split using SAM distances as primary criterion,
    pixel balance as secondary.
    
    Strategy:
    1. Find the two most spectrally distant classes → seeds
    2. Assign remaining classes to nearest seed
    3. This creates spectrally coherent groups
    """
    if len(classes) <= 1:
        return classes, []

    # Find the two most spectrally distant classes (seeds)
    max_dist  = -1
    seed_a, seed_b = classes[0], classes[1]

    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            ci, cj = classes[i], classes[j]
            d = sam_matrix[ci, cj]
            if d > max_dist:
                max_dist  = d
                seed_a, seed_b = ci, cj

    # Assign each remaining class to nearest seed
    group_a = [seed_a]
    group_b = [seed_b]

    for c in classes:
        if c == seed_a or c == seed_b:
            continue
        dist_a = sam_matrix[c, seed_a]
        dist_b = sam_matrix[c, seed_b]
        if dist_a <= dist_b:
            group_a.append(c)
        else:
            group_b.append(c)

    # Secondary: rebalance if one group is more than 3x the other by pixel count
    count_a = sum(pixel_counts.get(c, 0) for c in group_a)
    count_b = sum(pixel_counts.get(c, 0) for c in group_b)

    if count_a > count_b * 3 or count_b > count_a * 3:
        # Too imbalanced - fall back to hybrid approach
        # Move the largest class from the bigger group to smaller
        # only if it doesn't violate spectral coherence too much
        if count_a > count_b:
            biggest = max(group_a, key=lambda c: pixel_counts.get(c, 0))
            if biggest != seed_a:  # never move the seed
                group_a.remove(biggest)
                group_b.append(biggest)
        else:
            biggest = max(group_b, key=lambda c: pixel_counts.get(c, 0))
            if biggest != seed_b:
                group_b.remove(biggest)
                group_a.append(biggest)

    return group_a, group_b


def build_tree(classes, pixel_counts, sam_matrix, depth_mode='auto',
               max_depth=None, min_pixels=10, sam_threshold=0.05,
               current_depth=0):
    """
    Recursively build the RSSP tree.
    
    depth_mode:    'auto' | 'full' | int
    max_depth:     used when depth_mode is int
    min_pixels:    stop if minority class has fewer pixels
    sam_threshold: stop if classes are too spectrally similar
    
    Returns nested dict:
    {
        'classes': [...],
        'depth': int,
        'left':  {...} or None,
        'right': {...} or None
    }
    """
    node = {
        'classes': classes,
        'depth': current_depth,
        'left': None,
        'right': None
    }
    
    # ── Stopping conditions ───────────────────────────────────────────────────
    if len(classes) <= 2:
        return node
    
    min_class_pixels = min(pixel_counts.get(c, 0) for c in classes)
    if min_class_pixels < min_pixels:
        return node
    
    if depth_mode == 'auto' and current_depth >= 3:
        return node

    if depth_mode == 'auto':
        # Stop if all classes in this node are spectrally very similar
        sub_matrix = sam_matrix[np.ix_(classes, classes)]
        mean_sam = sub_matrix[sub_matrix > 0].mean() if (sub_matrix > 0).any() else 0
        if mean_sam < sam_threshold:
            return node
    
    elif depth_mode == 'full':
        pass  # never stop early, always recurse
    
    else:
        # depth_mode is an integer
        if current_depth >= depth_mode:
            return node
    
    # ── Split ─────────────────────────────────────────────────────────────────
    group_a, group_b = spectral_balanced_split(classes, pixel_counts, sam_matrix)
    
    if len(group_a) == 0 or len(group_b) == 0:
        return node
    
    node['left']  = build_tree(group_a, pixel_counts, sam_matrix,
                                depth_mode, max_depth, min_pixels,
                                sam_threshold, current_depth + 1)
    node['right'] = build_tree(group_b, pixel_counts, sam_matrix,
                                depth_mode, max_depth, min_pixels,
                                sam_threshold, current_depth + 1)
    
    return node


def build_rssp_tree(data, labels, num_classes, depth_mode='auto',
                    max_depth=None, min_pixels=10, sam_threshold=0.05):
    """
    Full pipeline: data → class means → continuum removal → SAM matrix → tree
    
    data:        (C, H, W)
    labels:      (H, W) numpy array
    num_classes: int (including background class 0)
    
    Returns: tree dict, sam_matrix, class_means
    """
    # Class ids excluding background
    classes = list(range(1, num_classes))
    
    # Pixel counts per class
    pixel_counts = {c: int((labels == c).sum()) for c in classes}
    
    print(f"Building RSSP tree for {len(classes)} classes")
    print(f"Pixel counts: { {c: pixel_counts[c] for c in classes} }")
    
    # Class mean spectra
    means = compute_class_means(data, labels, num_classes)  # (num_classes, C)
    
    # Continuum removal on means
    cr_means = continuum_removal_numpy(means[1:])  # skip background class 0
    
    # SAM distance matrix (only for labeled classes)
    sam_matrix_raw = compute_sam_matrix(cr_means)  # (num_classes-1, num_classes-1)
    
    # Pad to full size for indexing by class id
    sam_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    sam_matrix[1:, 1:] = sam_matrix_raw
    
    # Build tree
    tree = build_tree(classes, pixel_counts, sam_matrix,
                      depth_mode=depth_mode, max_depth=max_depth,
                      min_pixels=min_pixels, sam_threshold=sam_threshold)
    
    return tree, sam_matrix, means


def print_tree(node, indent=0):
    """Pretty print the tree structure."""
    prefix = "  " * indent
    classes = node['classes']
    print(f"{prefix}Node (depth {node['depth']}): classes {classes}")
    if node['left']:
        print(f"{prefix}  ├── LEFT:")
        print_tree(node['left'],  indent + 2)
    if node['right']:
        print(f"{prefix}  └── RIGHT:")
        print_tree(node['right'], indent + 2)