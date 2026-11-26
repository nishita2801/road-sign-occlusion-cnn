"""Evaluate a saved model on deterministic occluded GTSRB test images."""
import argparse
import torch
from data import make_loaders
from model import RoadSignCNN
from train import run_epoch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", default="data")
parser.add_argument("--checkpoint", default="checkpoints/road_sign_cnn.pt")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--workers", type=int, default=2)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = make_loaders(args.data_dir, args.batch_size, args.workers)
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
model = RoadSignCNN(checkpoint["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_state"])
metrics = run_epoch(model, test_loader, device)
print("GTSRB synthetic-occlusion test results")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
