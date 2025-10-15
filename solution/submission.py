from pathlib import Path
from typing import Callable, Optional
from xml.etree.ElementTree import Element, parse

import h5py
import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.tv_tensors import BoundingBoxes


class XMLLoader:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self):
        tree = parse(self.path)
        root = tree.getroot()
        return dict(self.xml_to_tv_tensor(element) for element in root.findall("image"))

    def xml_to_tv_tensor(self, element: Element) -> tuple[int, BoundingBoxes]:
        width = int(element.attrib["width"])
        height = int(element.attrib["height"])
        frame_index = int(element.attrib["name"].split(".")[0])
        frame_index = f"{self.path.stem}-{frame_index}"

        bbox_tensor = BoundingBoxes(
            [self.xml_to_bbox(xml_bbox) for xml_bbox in element],
            format="XYXY",
            canvas_size=(height, width),
            dtype=torch.float32,
        )
        return frame_index, bbox_tensor

    @staticmethod
    def xml_to_bbox(element: Element) -> list[float]:
        info_dict = element.attrib
        xmin = float(info_dict["xtl"])
        ymin = float(info_dict["ytl"])
        xmax = float(info_dict["xbr"])
        ymax = float(info_dict["ybr"])
        return [xmin, ymin, xmax, ymax]


class TokamDataset(VisionDataset):
    def __init__(
        self,
        root: Path = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        self.extract_annotations()
        self.load_images()

    def load_images(self):
        data_files = list(self.root.glob("*.h5"))
        self.images = []
        for file_path in data_files:
            with h5py.File(file_path) as f:
                data = f["density"]
                frame_indices = [
                    f'{file_path.stem}-{idx}' for idx in f["indices"]
                ]
                self.images.append(torch.tensor(
                    np.array(
                        [
                            np.expand_dims(image, axis=0)
                            for i, image in zip(frame_indices, data)
                            if self.annotations is None or i in self.annotations
                        ]
                    )
                ))
                if self.annotations is None:
                    self.idx_to_frame.extend(frame_indices)
        self.images = torch.cat(self.images, dim=0)
        self.num_frames = self.images.shape[0]
        self.image_width = self.images.shape[2]
        self.image_height = self.images.shape[1]

    def extract_annotations(self):
        try:
            label_files = list(self.root.glob("*.xml"))[-1]
            self.annotations = XMLLoader(label_files)()
            self.idx_to_frame = list(self.annotations.keys())
        except IndexError:
            self.annotations = None
            self.idx_to_frame = []
            print("No label files found")

    def __getitem__(self, index: int):
        frame_index = self.idx_to_frame[index]
        if self.annotations is not None:
            boxes = self.annotations[frame_index]
        else:
            boxes = []
        image = self.images[index].float()
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "frame_index": frame_index,
        }
        return image, target

    def __len__(self):
        return len(self.images)


def make_dataset(training_dir):
    return TokamDataset(training_dir)


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


def train_model(training_dir):
    train_dataset = make_dataset(training_dir)
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
