"""Train the custom CNN with joint sign and occlusion losses."""
import argparse
import random
from pathlib import Path
import torch
from torch import nn
from data import make_loaders
from model import RoadSignCNN


def run_epoch(model, loader, device, optimizer=None, occlusion_weight=1.0):
    training = optimizer is not None
    model.train(training)
    sign_loss_fn, occ_loss_fn = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "sign_correct": 0, "occ_tp": 0, "occ_fp": 0, "occ_fn": 0, "count": 0}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, occ in loader:
            images, labels, occ = images.to(device), labels.to(device), occ.to(device)
            outputs = model(images)
            loss = sign_loss_fn(outputs["sign_logits"], labels) + occlusion_weight * occ_loss_fn(outputs["occlusion_logits"], occ)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            sign_pred = outputs["sign_logits"].argmax(1)
            occ_pred = outputs["occlusion_logits"] >= 0
            totals["loss"] += loss.item() * labels.size(0)
            totals["sign_correct"] += (sign_pred == labels).sum().item()
            totals["occ_tp"] += ((occ_pred == 1) & (occ == 1)).sum().item()
            totals["occ_fp"] += ((occ_pred == 1) & (occ == 0)).sum().item()
            totals["occ_fn"] += ((occ_pred == 0) & (occ == 1)).sum().item()
            totals["count"] += labels.size(0)
    count = max(totals["count"], 1)
    precision = totals["occ_tp"] / max(totals["occ_tp"] + totals["occ_fp"], 1)
    recall = totals["occ_tp"] / max(totals["occ_tp"] + totals["occ_fn"], 1)
    return {"loss": totals["loss"] / count, "sign_accuracy": totals["sign_correct"] / count,
            "occlusion_precision": precision, "occlusion_recall": recall,
            "occlusion_f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="checkpoints/road_sign_cnn.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.workers)
    model = RoadSignCNN(num_classes=43).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_f1 = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; train images: {len(train_loader.dataset)}; test images: {len(test_loader.dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, test_loader, device)
        scheduler.step()
        print(f"{epoch:02d}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
              f"test_sign_acc={val_metrics['sign_accuracy']:.4f} occ_P={val_metrics['occlusion_precision']:.4f} "
              f"occ_R={val_metrics['occlusion_recall']:.4f} occ_F1={val_metrics['occlusion_f1']:.4f}")
        if val_metrics["occlusion_f1"] > best_f1:
            best_f1 = val_metrics["occlusion_f1"]
            torch.save({"model_state": model.state_dict(), "num_classes": 43, "image_size": 48,
                        "normalization_mean": [0.3403, 0.3121, 0.3214],
                        "normalization_std": [0.2724, 0.2608, 0.2669], "best_occlusion_f1": best_f1}, output)
    print(f"Best checkpoint saved to {output} (occlusion F1={best_f1:.4f})")

if __name__ == "__main__":
    main()
