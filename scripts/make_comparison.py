#!/usr/bin/env python3
"""Build labeled four-version and Eevee-versus-Cycles comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/System/Library/Fonts/SFNS.ttf")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", type=Path, default=PROJECT_DIR / "renders" / "01_soft_staged.png")
    parser.add_argument("--hdri", type=Path, default=PROJECT_DIR / "renders" / "02_hdri_only.png")
    parser.add_argument("--instant", type=Path, default=PROJECT_DIR / "renders" / "03_instant_print.png")
    parser.add_argument("--cycles", type=Path, default=PROJECT_DIR / "renders" / "04_instant_print_cycles.png")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "renders" / "comparison.png")
    parser.add_argument("--engine-output", type=Path, default=PROJECT_DIR / "renders" / "instant_eevee_vs_cycles.png")
    return parser.parse_args()


def make_grid(items: list[tuple[str, Image.Image]], columns: int) -> Image.Image:
    panel_size = 1024
    gap = 32
    label_height = 96
    cell_height = panel_size + label_height
    rows = (len(items) + columns - 1) // columns
    background = (28, 26, 24)
    canvas = Image.new(
        "RGB",
        (panel_size * columns + gap * (columns - 1), cell_height * rows + gap * (rows - 1)),
        background,
    )
    font = ImageFont.truetype(str(FONT_PATH), 38)
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        column = index % columns
        row = index // columns
        x0 = column * (panel_size + gap)
        y0 = row * (cell_height + gap)
        panel = image.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x0, y0 + label_height))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        x = x0 + (panel_size - text_width) // 2
        y = y0 + (label_height - (box[3] - box[1])) // 2 - box[1]
        draw.text((x, y), label, fill=(238, 232, 222), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    soft = Image.open(args.soft).convert("RGB")
    hdri = Image.open(args.hdri).convert("RGB")
    instant = Image.open(args.instant).convert("RGB")
    cycles = Image.open(args.cycles).convert("RGB")
    sizes = {soft.size, hdri.size, instant.size, cycles.size}
    if len(sizes) != 1:
        raise ValueError(f"Render dimensions differ: {sorted(sizes)}")

    canvas = make_grid([
        ("SOFT STAGED", soft),
        ("HDRI ONLY", hdri),
        ("INSTANT PRINT — EEVEE", instant),
        ("INSTANT PRINT — CYCLES", cycles),
    ], columns=2)
    engine_canvas = make_grid([
        ("INSTANT PRINT — EEVEE", instant),
        ("INSTANT PRINT — CYCLES", cycles),
    ], columns=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format="PNG", optimize=True)
    engine_canvas.save(args.engine_output, format="PNG", optimize=True)

    print(f"Wrote {args.output} ({canvas.width}x{canvas.height})")
    print(f"Wrote {args.engine_output} ({engine_canvas.width}x{engine_canvas.height})")


if __name__ == "__main__":
    main()
