import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

EVAL_SETS = ["test", "private_test"]


def compute_ap(predictions, targets):
    all_ap = []
    for frame_index in targets.keys():
        if frame_index in predictions:
            y, y_pred = targets[frame_index], predictions[frame_index]
            if len(y_pred) == 0 and len(y) == 0:
                iou_finale = 1.0
            elif len(y_pred) == 0 or len(y) == 0:
                iou_finale = 0.0
            else:
                boxes = torch.tensor([yp[:4] for yp in y_pred])
                scores = torch.tensor([yp[4] for yp in y_pred])
                y = torch.tensor(y)[:, :4]
                ious = box_iou(y, boxes)
                ious = ious.numpy() * scores.numpy()
                row_ind, col_ind = linear_sum_assignment(ious, maximize=True)
                iou_finale = ious[row_ind, col_ind].sum()

            all_ap.append(iou_finale / max(len(y), len(y_pred)))
        else:
            if len(targets[frame_index]) == 0:
                all_ap.append(1.0)
            else:
                all_ap.append(0.0)
    return float(np.mean(all_ap))


def read_xml(path):
    from xml.etree.ElementTree import parse

    tree = parse(path)
    root = tree.getroot()
    annotations = {}
    for element in root.findall("image"):
        if "index" in element.attrib:
            frame_index = element.attrib["index"].split("-")[1]
        else:
            frame_index = element.attrib["name"].split(".")[0]
        annotations[frame_index] = [
            (
                float(bbox.attrib["xtl"]),
                float(bbox.attrib["ytl"]),
                float(bbox.attrib["xbr"]),
                float(bbox.attrib["ybr"]),
                float(bbox.attrib.get("score", 1.0)),
            )
            for bbox in element.findall("box")
        ]
    return annotations


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f'Scoring {eval_set}')

        predictions = read_xml(prediction_dir / f'{eval_set}_predictions.xml')
        targets = read_xml(reference_dir / f'{eval_set}_labels.xml')
        scores[eval_set] = compute_ap(predictions, targets)

    # Add train and test times in the score
    json_durations = (prediction_dir / 'metadata.json').read_text()
    durations = json.loads(json_durations)
    scores.update(**durations)
    print(scores)

    # Write output scores
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scores.json').write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )
