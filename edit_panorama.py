#!/usr/bin/env python3
"""Wrap-pad a panorama with a slice of its own opposite edge before editing,
then trim the padding back off afterward.

Instead of duplicating the whole image (which confuses edit models into
thinking there are two of everything), this takes the last P% of the image
and prepends it, and the first P% and appends it - so the model sees real
wraparound context at both edges without doubling the whole scene. Total
width becomes (100 + 2*P)% of the original.

Usage:
    python edit_panorama.py stitch inference/inputs/Street_View_360_1.jpg --percent 10
    python edit_panorama.py split inference/outputs/Street_View_360_1_stitched_10_edited.png --percent 10

--percent defaults to 10 if omitted. Use the same --percent value for both
stitch and split so the trim matches what was padded.
"""
import argparse
from pathlib import Path

from PIL import Image


def _fmt_percent(percent: float) -> str:
    return f"{percent:g}"


def stitch(input_path: Path, percent: float = 10.0) -> Path:
    img = Image.open(input_path)
    w, h = img.size
    strip_w = round(w * percent / 100)

    left_strip = img.crop((w - strip_w, 0, w, h))
    right_strip = img.crop((0, 0, strip_w, h))

    out = Image.new(img.mode, (w + 2 * strip_w, h))
    out.paste(left_strip, (0, 0))
    out.paste(img, (strip_w, 0))
    out.paste(right_strip, (strip_w + w, 0))

    out_path = input_path.with_name(
        f"{input_path.stem}_stitched_{_fmt_percent(percent)}{input_path.suffix}"
    )
    out.save(out_path)
    return out_path


def split(input_path: Path, percent: float = 10.0, crop_type: str = "center") -> list[Path]:
    """Crop the original-width (2:1) window back out of a padded panorama.

    The padding adds `strip_w` columns on each side, so there are three
    natural places to take the original-width window from: flush left
    (offset 0), flush right (offset total_w - original_w), or centered
    (offset strip_w, i.e. discarding equal padding from both sides - the
    old default behavior). `crop_type="all"` returns all three so you can
    compare which one has the cleanest wraparound seam.
    """
    img = Image.open(input_path)
    total_w, h = img.size

    original_w = round(total_w / (1 + 2 * percent / 100))
    strip_w = round((total_w - original_w) / 2)

    offsets = {
        "left": 0,
        "center": strip_w,
        "right": total_w - original_w,
    }
    types = ["left", "center", "right"] if crop_type == "all" else [crop_type]

    out_paths = []
    for t in types:
        cropped = img.crop((offsets[t], 0, offsets[t] + original_w, h))
        out_path = input_path.with_name(
            f"{input_path.stem}_split_{_fmt_percent(percent)}_{t}{input_path.suffix}"
        )
        cropped.save(out_path)
        out_paths.append(out_path)
    return out_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_stitch = sub.add_parser("stitch", help="pad a panorama with P%% of its own wraparound edge on each side")
    p_stitch.add_argument("input", type=Path)
    p_stitch.add_argument("--percent", "-p", type=float, default=10.0, help="percent of width to pad on each side (default 10)")

    p_split = sub.add_parser("split", help="trim the wraparound padding back off an edited panorama")
    p_split.add_argument("input", type=Path)
    p_split.add_argument("--percent", "-p", type=float, default=10.0, help="percent that was used when stitching (default 10)")
    p_split.add_argument("--type", choices=["left", "right", "center", "all"], default="center", help="which 2:1 window to crop out (default center)")

    args = parser.parse_args()

    if args.mode == "stitch":
        out_path = stitch(args.input, args.percent)
        print(f"Saved {out_path}")
    else:
        for out_path in split(args.input, args.percent, args.type):
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
