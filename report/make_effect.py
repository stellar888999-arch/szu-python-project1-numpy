"""Generate the report's 16-coordinate comparison from the actual NPZ export."""
from pathlib import Path
import argparse
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("npz", nargs="?", type=Path,
                    default=ROOT / "evidence/output_tokens.npz")
args = parser.parse_args()
with np.load(args.npz, allow_pickle=False) as data:
    ids = data["ids"].tolist()
    values = data["output_tokens"][[ids.index("image_01"), ids.index("image_02")], :16]
limit = max(0.05, math.ceil(float(np.abs(values).max()) / 0.05) * 0.05)
lines = [r"\begin{tikzpicture}[x=.73cm,y=1.05cm,font=\small]"]
for tick in (-1, 0, 1):
    label = f"{limit*tick:.2f}"
    lines.append(rf"\draw[gray!35] (1,{tick}) -- (16.25,{tick});")
    lines.append(rf"\node[left] at (1,{tick}) {{{label}}};")
lines += [r"\draw[->] (1,-1.1) -- (1,1.28);",
          r"\draw[->] (1,-1.1) -- (16.5,-1.1);",
          r"\node[anchor=south west] at (1,1.29) {特征值};",
          r"\node[below] at (8.5,-1.55) {特征下标};"]
for x in (1, 4, 8, 12, 16):
    lines.append(rf"\node[below] at ({x},-1.13) {{{x}}};")
for row, color, style, y, label in zip(
    values, ("blue!70!black", "orange!80!black"), ("solid", "dashed"),
    (1.52, 1.52), (r"image\_01", r"image\_02"),
):
    points = " ".join(f"({i},{float(v)/limit:.6f})" for i, v in enumerate(row, 1))
    lines.append(rf"\draw[{color},{style},thick] plot coordinates {{{points}}};")
    x = 6 if style == "solid" else 11
    lines.append(rf"\draw[{color},{style},thick] ({x},{y}) -- ++(1,0) node[right] {{\ttfamily {label}}};")
lines.append(r"\end{tikzpicture}")
(ROOT / "report/effect.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
