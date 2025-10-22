from pathlib import Path
from typing import Callable, Optional

import h5py
import torch
import numpy as np
from torchvision.datasets.vision import VisionDataset


from .xml_loader import XMLLoader


class TokamDataset(VisionDataset):
    def __init__(
        self,
        root: Path = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        include_unlabeled: bool = False,  # TODO: implement this
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms
        )
        self.include_unlabeled = include_unlabeled

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
                            if (
                                self.annotations is None
                                or self.include_unlabeled
                                or i in self.annotations
                            )
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
        if self.annotations is not None and frame_index in self.annotations:
            boxes = self.annotations[frame_index]
        else:
            boxes = []
        image = self.images[index].float()
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "frame_index": frame_index,
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.images)
