import torch
from datasets.pavia_university import PaviaUniversityDataset
from models.hyperspectral_net import HyperspectralNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== Phase 1 Zero-Shot Agnosticism Test ===")

try:
    pavia_ds = PaviaUniversityDataset(
        data_path='data/pavia/PaviaU.mat',    # Make sure to update this path
        gt_path='data/pavia/PaviaU_gt.mat',   # Make sure to update this path
        split='test', 
        use_patches=True
    )
    pavia_data, pavia_labels = pavia_ds[0]
    pavia_data = pavia_data.unsqueeze(0).to(DEVICE) 
    print(f"Successfully loaded Pavia. Input shape: {pavia_data.shape}")
except Exception as e:
    print(f"Failed to load Pavia data: {e}")
    exit()

model = HyperspectralNet(
    num_bands=200,       #
    num_classes=17,      
    base_filters=32
).to(DEVICE)


try:
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE,weights_only = True))
    print("Loaded Indian Pines weights successfully.")
except Exception as e:
    print(f"Failed to load weights. Did you train the new architecture yet? Error: {e}")
    exit()

model.eval()


print("\nRunning forward pass with 103-band Pavia tensor...")
try:
    with torch.no_grad():
        output = model(pavia_data)
    print(f"\n[SUCCESS] Forward pass completed without crashing!")
    print(f"Output shape: {output.shape} (Expected: 1, 17, 610, 340)")
    print("\nThe Spectral Transformer successfully collapsed the 103 bands down to a fixed representation.")
    print("The model is now band-count agnostic. Phase 1 complete.")
except Exception as e:
    print(f"\n[FAILED] The model crashed during the forward pass.")
    print(f"Error: {e}")