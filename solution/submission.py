import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from tokam2d_utils import TokamDataset


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


def train_model(training_dir):
    train_dataset = TokamDataset(training_dir)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, collate_fn=collate_fn
    )

    model = fasterrcnn_resnet50_fpn()
    model.train()

    optimizer = torch.optim.SGD(model.parameters())

    max_epochs = 1

    for _ in range(max_epochs):
        for images, targets in train_dataloader:
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            full_loss = sum(loss for loss in loss_dict.values())
            full_loss.backward()
            optimizer.step()
            break

    model.eval()
    return model
