"""Export norm_stats.npz to a C header file for STM32 firmware."""
import numpy as np
from pathlib import Path

stats = np.load(Path(__file__).parent / "features" / "norm_stats.npz")
mean = stats["mean"]
std = stats["std"]

header_path = Path(__file__).parent.parent / "STM32" / "EmbeddedAI" / "Core" / "Inc" / "norm_stats.h"

with open(header_path, "w") as f:
    f.write("/* Auto-generated from norm_stats.npz — DO NOT EDIT */\n")
    f.write("#ifndef NORM_STATS_H\n#define NORM_STATS_H\n\n")
    f.write(f"#define N_MFCC {len(mean)}\n\n")

    f.write("static const float mfcc_mean[N_MFCC] = {\n  ")
    f.write(",\n  ".join(f"{v:.8f}f" for v in mean))
    f.write("\n};\n\n")

    f.write("static const float mfcc_std[N_MFCC] = {\n  ")
    f.write(",\n  ".join(f"{v:.8f}f" for v in std))
    f.write("\n};\n\n")

    f.write("#endif /* NORM_STATS_H */\n")

print(f"Written {header_path}")
print(f"  N_MFCC = {len(mean)}")
print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}]")
print(f"  std  range: [{std.min():.4f}, {std.max():.4f}]")
