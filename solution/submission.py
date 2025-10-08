from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class TokamDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path] = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            self,
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )


def make_dataset(training_dir):
    data_files = list(training_dir.glob("*.h5"))
    label_files = list(training_dir.glob("*.xml"))
    train_data = ...  # Filter out un-annotated data
    labels = ...  # Create bounding boxes
    return BoxDataset(train_data, labels)


def train_model(training_dir):
    train_data = make_dataset(training_dir)
    train_dataloader = torch.dataset.DataLoader(train_data, batch_size=4)

    model = fasterrcnn_resnet50_fpn()
    model.train()

    optimizer = torch.optim.SGD(model.parameters())

    for _ in range(max_epochs):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            loss_dict = model(X, y)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.step()

    model.eval()
    return model
