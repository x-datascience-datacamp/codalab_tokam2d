import json
import sys
import time
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree

import torch

EVAL_SETS = ["test", "private_test"]


def dump_to_xml(y_pred, filepath):
    root = Element("predictions")
    for y_p in y_pred:
        frame_idx = y_p["frame_index"]
        frame_elem = Element("image", index=str(frame_idx))
        for box, score in zip(y_p['boxes'], y_p['scores']):
            box_elem = Element(
                "box", score=str(score.item()),
                xtl=str(box[0].item()), ytl=str(box[1].item()),
                xbr=str(box[2].item()), ybr=str(box[3].item()),
            )
            frame_elem.append(box_elem)
        root.append(frame_elem)
    tree = ElementTree(root)
    tree.write(filepath)


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


def evaluate_model(model, data_dir):
    from tokam2d_utils import TokamDataset

    eval_dataset = TokamDataset(data_dir)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=2, collate_fn=collate_fn
    )

    model.eval()
    res = []
    for X, y in eval_dataloader:
        with torch.no_grad():
            y_pred = model(X)

        # Add back frame index
        y_pred = [
            {**y_p, "frame_index": y_t["frame_index"]}
            for y_p, y_t in zip(y_pred, y)
        ]
        # Check how to make this work
        res.extend(y_pred)

    return res


def main(data_dir, output_dir):
    from submission import train_model

    training_dir = data_dir / "train"

    print("Training the model")
    start = time.time()
    model = train_model(training_dir)
    train_time = time.time() - start
    print("-" * 10)
    print("Evaluate the model")
    start = time.time()
    res = {}
    for eval_set in EVAL_SETS:
        res[eval_set] = evaluate_model(model, data_dir / eval_set)
    test_time = time.time() - start
    print("-" * 10)
    duration = train_time + test_time
    print(f"Completed Prediction. Total duration: {duration}")

    # Write output files
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w+") as f:
        json.dump(dict(train_time=train_time, test_time=test_time), f)
    for eval_set in EVAL_SETS:
        filepath = output_dir / f"{eval_set}_predictions.xml"
        dump_to_xml(res[eval_set], filepath)
    print()
    print("Ingestion Program finished. Moving on to scoring")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for codabench"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="",
    )

    args = parser.parse_args()
    sys.path.append(args.submission_dir)
    sys.path.append(Path(__file__).parent.resolve())

    main(Path(args.data_dir), Path(args.output_dir))
