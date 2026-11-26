"""GTSRB loaders and synthetic occlusion augmentation."""

import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import GTSRB
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class OccludedGTSRB(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 48,
        download: bool = True,
        occlusion_probability: float = 0.7,
        max_occluders: int = 2,
    ):
        # torchvision expects lowercase split names: "train" or "test".
        split = split.lower()
        self.base = GTSRB(root=root, split=split, download=download)
        self.size = image_size
        self.occlusion_probability = occlusion_probability
        self.max_occluders = max_occluders

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]

        image = TF.resize(
            image,
            [self.size, self.size],
            interpolation=InterpolationMode.BILINEAR,
        )
        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            [0.3403, 0.3121, 0.3214],
            [0.2724, 0.2608, 0.2669],
        )

        occluded = int(random.random() < self.occlusion_probability)

        if occluded:
            image = image.clone()
            height, width = image.shape[-2:]

            for _ in range(random.randint(1, self.max_occluders)):
                patch_height = random.randint(
                    max(3, height // 8),
                    max(4, height // 2),
                )
                patch_width = random.randint(
                    max(3, width // 8),
                    max(4, width // 2),
                )
                y = random.randint(0, height - patch_height)
                x = random.randint(0, width - patch_width)

                color = torch.tensor(
                    random.choice(
                        [
                            (0.20, 0.28, 0.12),
                            (0.32, 0.22, 0.12),
                            (0.36, 0.35, 0.32),
                        ]
                    ),
                    dtype=image.dtype,
                ).view(3, 1, 1)

                alpha = random.uniform(0.65, 0.95)
                image[:, y:y + patch_height, x:x + patch_width] = (
                    alpha * color
                    + (1 - alpha)
                    * image[:, y:y + patch_height, x:x + patch_width]
                )

        return image, int(label), float(occluded)


def make_loaders(
    data_dir: str,
    batch_size: int = 128,
    workers: int = 2,
    image_size: int = 48,
):
    train_dataset = OccludedGTSRB(
        data_dir,
        "train",
        image_size=image_size,
        occlusion_probability=0.7,
    )
    test_dataset = OccludedGTSRB(
        data_dir,
        "test",
        image_size=image_size,
        occlusion_probability=0.5,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader