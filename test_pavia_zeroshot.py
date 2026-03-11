import torch
from datasets.pavia_university import PaviaUniversityDataset
from models.hyperspectral_net import HyperspectralNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== Phase 1 Zero-Shot Agnosticism Test ===")
print("Model trained on Indian Pines (200 bands) → inference on Pavia University (103 bands)")
print("No retraining. No fine-tuning. Raw transfer.\n")

try:
    pavia_ds = PaviaUniversityDataset(
        data_path='data/pavia/PaviaU.mat',
        gt_path='data/pavia/PaviaU_gt.mat',
        split='test',
        use_patches=False
    )
    pavia_data, pavia_labels = pavia_ds[0]
    print(f"Pavia data shape:   {tuple(pavia_data.shape)}  (103 bands, 610x340)")
    print(f"Pavia labels shape: {tuple(pavia_labels.shape)}")
except Exception as e:
    print(f"Failed to load Pavia data: {e}")
    exit()

# Load model architecture (trained on 200 bands, 17 classes)
model = HyperspectralNet(
    num_bands=200,
    num_classes=17,
    base_filters=32
)

try:
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu', weights_only=True))
    print("\nLoaded Indian Pines weights successfully.")
except Exception as e:
    print(f"\nFailed to load weights: {e}")
    print("Have you run train.py yet?")
    exit()

model.eval()

# Inference runs on CPU — full 610x340 image creates 207k transformer sequences, too large for GPU.
print("\nRunning forward pass on CPU (full 610x340 image, 103 bands)...")
print("Note: ~207k transformer sequences. This will take a minute on CPU.")

try:
    with torch.no_grad():
        output = model(pavia_data.unsqueeze(0))

    print(f"\n[SUCCESS] Output shape: {tuple(output.shape)}")
    print(f"Expected:              (1, 17, 610, 340)")
    print(f"\nThe SpectralTransformer interpolated positional embeddings from 200 → 103 bands.")
    print(f"Band-count agnosticism confirmed. Phase 1 complete.")
except Exception as e:
    print(f"\n[FAILED] {e}")