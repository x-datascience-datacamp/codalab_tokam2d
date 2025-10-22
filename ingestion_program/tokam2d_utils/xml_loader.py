from pathlib import Path
from xml.etree.ElementTree import Element, parse

import torch
from torchvision.tv_tensors import BoundingBoxes


class XMLLoader:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self):
        tree = parse(self.path)
        root = tree.getroot()
        return dict(
            self.xml_to_tv_tensor(element) for element in root.findall("image")
        )

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
