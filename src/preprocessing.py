# Crops each annotated defect out of the raw PCB photos and builds a
# classification dataset from it (data/raw -> data/processed + data/splits).
#
# Run:
#   python -m src.preprocessing

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASS_ALIASES = {
    "missing_hole": "missing_hole",
    "missing hole": "missing_hole",
    "mouse_bite": "mouse_bite",
    "mouse bite": "mouse_bite",
    "open_circuit": "open_circuit",
    "open circuit": "open_circuit",
    "short": "short",
    "spur": "spur",
    "spurious_copper": "spurious_copper",
    "spurious copper": "spurious_copper",
}

ALL_KNOWN_CLASSES = sorted(set(CLASS_ALIASES.values()))
DEFAULT_TARGET_CLASSES = ["missing_hole", "spur", "short", "open_circuit"]


def normalize_class_name(raw_name):
    key = raw_name.strip().lower()
    if key not in CLASS_ALIASES:
        raise ValueError(f"Unrecognized defect class name '{raw_name}'.")
    return CLASS_ALIASES[key]


@dataclass
class BBoxAnnotation:
    source_image: str
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def parse_voc_annotation(xml_path, image_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if filename is None:
        raise ValueError(f"Annotation {xml_path} has no <filename> element")

    image_path = image_dir / filename
    if not image_path.exists():
        # a few files have a filename that doesn't quite match the image on disk
        candidates = list(image_dir.glob(f"{Path(filename).stem}.*"))
        if not candidates:
            candidates = [
                p for p in image_dir.iterdir()
                if p.stem.lower() == Path(filename).stem.lower()
            ]
        if not candidates:
            raise FileNotFoundError(
                f"Annotation {xml_path} references image '{filename}' "
                f"which was not found in {image_dir}"
            )
        image_path = candidates[0]

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        try:
            class_name = normalize_class_name(name)
        except ValueError:
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = int(round(float(bnd.findtext("xmin"))))
        ymin = int(round(float(bnd.findtext("ymin"))))
        xmax = int(round(float(bnd.findtext("xmax"))))
        ymax = int(round(float(bnd.findtext("ymax"))))
        boxes.append(BBoxAnnotation(str(image_path), class_name, xmin, ymin, xmax, ymax))
    return boxes


def collect_annotations(raw_dir, target_classes):
    annotations_root = raw_dir / "Annotations"
    images_root = raw_dir / "images"
    if not annotations_root.exists() or not images_root.exists():
        raise FileNotFoundError(
            f"Expected '{annotations_root}' and '{images_root}' to exist. "
            "Place the raw PKU-Market-PCB dataset under data/raw/ first."
        )

    boxes = []
    for class_dir in sorted(p for p in annotations_root.iterdir() if p.is_dir()):
        image_dir = images_root / class_dir.name
        if not image_dir.exists():
            continue
        for xml_path in sorted(class_dir.glob("*.xml")):
            file_boxes = parse_voc_annotation(xml_path, image_dir)
            boxes.extend(b for b in file_boxes if b.class_name in target_classes)
    return boxes


def crop_with_margin(image, box, margin_frac):
    # crop the box with a bit of extra space around it, so the model sees
    # some surrounding trace/board context and not just the tight defect
    width, height = image.size
    box_w = box.xmax - box.xmin
    box_h = box.ymax - box.ymin
    pad_x = max(1, int(round(box_w * margin_frac)))
    pad_y = max(1, int(round(box_h * margin_frac)))

    left = max(0, box.xmin - pad_x)
    top = max(0, box.ymin - pad_y)
    right = min(width, box.xmax + pad_x)
    bottom = min(height, box.ymax + pad_y)
    return image.crop((left, top, right, bottom))


def process_crop(image, image_size, grayscale):
    image = image.convert("L") if grayscale else image.convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    return np.array(image, dtype=np.uint8)


def split_source_images(source_images_by_class, val_frac, test_frac, seed):
    # split by *board* (source image), not by crop - several crops come from
    # the same board, so splitting per-crop would leak the same board into
    # both train and test
    rng = np.random.default_rng(seed)
    assignment = {}
    for class_name, images in source_images_by_class.items():
        images = sorted(set(images))
        rng.shuffle(images)
        n = len(images)
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - n_val - 1))
        val_set = images[:n_val]
        test_set = images[n_val : n_val + n_test]
        train_set = images[n_val + n_test :]
        for img in train_set:
            assignment[img] = "train"
        for img in val_set:
            assignment[img] = "val"
        for img in test_set:
            assignment[img] = "test"
    return assignment


def run_preprocessing(raw_dir, output_dir, splits_dir, target_classes, image_size,
                       grayscale, margin, val_frac, test_frac, seed):
    boxes = collect_annotations(raw_dir, target_classes)
    if not boxes:
        raise RuntimeError("No annotated defects found for the requested classes.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in target_classes:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    image_cache = {}
    rows = []
    per_image_counters = {}
    boxes_per_image = {}
    for box in boxes:
        boxes_per_image[box.source_image] = boxes_per_image.get(box.source_image, 0) + 1

    for box in boxes:
        if box.source_image not in image_cache:
            image_cache[box.source_image] = Image.open(box.source_image)
        image = image_cache[box.source_image]

        crop = crop_with_margin(image, box, margin)
        if crop.size[0] == 0 or crop.size[1] == 0:
            continue
        array = process_crop(crop, image_size, grayscale)

        stem = Path(box.source_image).stem
        idx = per_image_counters.get(box.source_image, 0)
        per_image_counters[box.source_image] = idx + 1
        crop_filename = f"{stem}_{idx:02d}.png"
        crop_path = output_dir / box.class_name / crop_filename
        Image.fromarray(array).save(crop_path)

        rows.append({"crop_path": str(crop_path), "class_name": box.class_name, "source_image": box.source_image})

        # don't hold every full-res board photo in memory at once
        if per_image_counters[box.source_image] >= boxes_per_image[box.source_image]:
            image_cache.pop(box.source_image, None)

    manifest = pd.DataFrame(rows)
    split_assignment = split_source_images(
        {c: manifest.loc[manifest["class_name"] == c, "source_image"].tolist() for c in target_classes},
        val_frac,
        test_frac,
        seed,
    )
    manifest["split"] = manifest["source_image"].map(split_assignment)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split_df = manifest[manifest["split"] == split_name][["crop_path", "class_name", "source_image"]]
        split_df.to_csv(splits_dir / f"{split_name}.csv", index=False)
    manifest.to_csv(splits_dir / "manifest.csv", index=False)
    return manifest


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--classes", nargs="+", default=DEFAULT_TARGET_CLASSES, choices=ALL_KNOWN_CLASSES)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--grayscale", action="store_true", default=True)
    parser.add_argument("--rgb", dest="grayscale", action="store_false")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_arg_parser().parse_args()
    manifest = run_preprocessing(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        target_classes=args.classes,
        image_size=args.image_size,
        grayscale=args.grayscale,
        margin=args.margin,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(f"Wrote {len(manifest)} crops across {manifest['class_name'].nunique()} classes.")
    print(manifest.groupby(["class_name", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
