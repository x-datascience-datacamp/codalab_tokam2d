from pathlib import Path
from typing import Callable, Optional
from xml.etree.ElementTree import Element

import h5py
import torch
from defusedxml.ElementTree import parse
from torchvision.datasets.vision import VisionDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.tv_tensors import BoundingBoxes


class XMLLoader:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self):
        tree = parse(self.path)
        root = tree.getroot()
        return dict(self.xml_to_tv_tensor(element for element in root.findall("image")))

    def xml_to_tv_tensor(self, element: Element) -> tuple[int, BoundingBoxes]:
        width = int(element.attrib["width"])
        height = int(element.attrib["height"])
        frame_index = int(element.attrib["name"].split(".")[0])

        bbox_tensor = BoundingBoxes(
            [self.xml_to_bbox(xml_bbox) for xml_bbox in element],
            format="XYXY",
            canvas_size=(height, width),
            dtype=torch.float32,
        )
        return frame_index, bbox_tensor

    @staticmethod
    def xml_to_bbox(self, element: Element) -> torch.Tensor:
        info_dict = element.attrib
        xmin = float(info_dict["xtl"])
        ymin = float(info_dict["ytl"])
        xmax = float(info_dict["xbr"])
        ymax = float(info_dict["ybr"])
        return torch.tensor([xmin, ymin, xmax, ymax])


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
        file_path = data_files[-1]
        with h5py.File(file_path) as f:
            self.images = torch.tensor(
                [image for i, image in f["density"] if i in self.annotation_dict]
            )

        self.num_frames = self.images.shape[0]
        self.image_width = self.images.shape[2]
        self.image_height = self.images.shape[1]

    def extract_annotations(self):
        label_files = list(self.root.glob("*.xml"))
        self.annotation_dict = XMLLoader(label_files)()
        self.annotations = torch.tensor(self.annotation_dict.values())

    def __getitem__(self, index: int):
        return self.images[index], self.annotations[index]

    def __len__(self):
        return len(self.images)


def make_dataset(training_dir):
    print()
    print(training_dir)
    print()
    return TokamDataset(training_dir)


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
